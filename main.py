import os
import time
import threading
import cv2
import torch
from ultralytics import YOLO

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Drag & Drop support (optional but recommended)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_OK = True
except Exception:
    DND_OK = False


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class App(ctk.CTk if not DND_OK else TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Face Shield AI (POWEREN)")
        self.geometry("940x900")
        self.minsize(940, 600)

        self.running = False
        self.stop_flag = False

        # -----------------------------
        # UI variables
        # -----------------------------
        self.video_path = ctk.StringVar(value="")
        self.model_path = ctk.StringVar(value="yolov8l_100e.pt")
        self.output_path = ctk.StringVar(value="faces_blurred.mp4")

        self.conf = ctk.DoubleVar(value=0.20)
        self.iou = ctk.DoubleVar(value=0.45)
        self.imgsz = ctk.IntVar(value=1280)
        self.blur_strength = ctk.IntVar(value=75)

        self.use_half = ctk.BooleanVar(value=True)
        self.preview = ctk.BooleanVar(value=True)

        # -----------------------------
        # Layout
        # -----------------------------
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=12)
        header.grid(row=0, column=0, padx=16, pady=(16, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Face Shield AI (POWEREN)",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.grid(row=0, column=0, padx=16, pady=(12, 0), sticky="w")

        note = ctk.CTkLabel(
            header,
            text="Powered by poweren.ir",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        )
        note.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        body = ctk.CTkFrame(self, corner_radius=12)
        body.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # Left panel
        left = ctk.CTkFrame(body, corner_radius=12)
        left.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)

        # Right panel
        right = ctk.CTkFrame(body, corner_radius=12)
        right.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        # -----------------------------
        # Video drop zone
        # -----------------------------
        drop = ctk.CTkFrame(left, corner_radius=12)
        drop.grid(row=0, column=0, padx=12, pady=12, sticky="ew")
        drop.grid_columnconfigure(0, weight=1)

        self.drop_label = ctk.CTkLabel(
            drop,
            text="Drop Video Here\n(or click 'Browse Video')",
            height=120,
            justify="center",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.drop_label.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")

        helper_drop = ctk.CTkLabel(
            drop,
            text="Tip: Drag & drop a .mp4/.avi/.mkv file to auto-fill the input path.",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        )
        helper_drop.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

        if DND_OK:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.drop_label.configure(text="Drag & Drop needs tkinterdnd2\n(or click 'Browse Video')")

        browse_btn = ctk.CTkButton(drop, text="Browse Video", command=self.browse_video)
        browse_btn.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")

        self.video_path_lbl = ctk.CTkLabel(left, textvariable=self.video_path, wraplength=420, anchor="w")
        self.video_path_lbl.grid(row=1, column=0, padx=12, pady=(0, 6), sticky="ew")

        helper_video = ctk.CTkLabel(
            left,
            text="Selected input video path will appear here.",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        )
        helper_video.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="w")

        # -----------------------------
        # Model selection
        # -----------------------------
        model_box = ctk.CTkFrame(left, corner_radius=12)
        model_box.grid(row=3, column=0, padx=12, pady=6, sticky="ew")
        model_box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(model_box, text="Model (.pt)", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=12, pady=(10, 2), sticky="w"
        )
        ctk.CTkLabel(
            model_box,
            text="Your YOLO face model weights file (trained for face detection).",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        ).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        model_row = ctk.CTkFrame(model_box, fg_color="transparent")
        model_row.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="ew")
        model_row.grid_columnconfigure(0, weight=1)

        self.model_entry = ctk.CTkEntry(model_row, textvariable=self.model_path)
        self.model_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(model_row, text="Browse", width=90, command=self.browse_model).grid(row=0, column=1)

        # -----------------------------
        # Output selection
        # -----------------------------
        out_box = ctk.CTkFrame(left, corner_radius=12)
        out_box.grid(row=4, column=0, padx=12, pady=6, sticky="ew")
        out_box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(out_box, text="Output Path", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=12, pady=(10, 2), sticky="w"
        )
        ctk.CTkLabel(
            out_box,
            text="Where the blurred video will be saved (MP4/AVI).",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        ).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        out_row = ctk.CTkFrame(out_box, fg_color="transparent")
        out_row.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="ew")
        out_row.grid_columnconfigure(0, weight=1)

        self.out_entry = ctk.CTkEntry(out_row, textvariable=self.output_path)
        self.out_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(out_row, text="Save As", width=90, command=self.save_as).grid(row=0, column=1)

        # -----------------------------
        # Controls (right panel)
        # -----------------------------
        controls = ctk.CTkFrame(right, corner_radius=12)
        controls.grid(row=0, column=0, padx=12, pady=12, sticky="ew")
        controls.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(controls, text="Controls", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 6), sticky="w"
        )

        self._slider_row(
            controls,
            label="Confidence (CONF)",
            helper="Lower = detects more faces (may add false positives). Higher = stricter.",
            var=self.conf,
            from_=0.05,
            to=0.80,
            row=1
        )

        self._slider_row(
            controls,
            label="IOU",
            helper="Controls box merging. Usually keep around 0.45.",
            var=self.iou,
            from_=0.10,
            to=0.80,
            row=2
        )

        ctk.CTkLabel(controls, text="Image Size (IMGSZ)", font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, padx=12, pady=(12, 2), sticky="w"
        )
        ctk.CTkLabel(
            controls,
            text="Bigger = better accuracy but slower. Recommended: 960 or 1280.",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        ).grid(row=4, column=0, padx=12, pady=(0, 6), sticky="w")

        self.imgsz_combo = ctk.CTkOptionMenu(controls, values=["640", "960", "1280"], command=self.on_imgsz_change)
        self.imgsz_combo.set(str(self.imgsz.get()))
        self.imgsz_combo.grid(row=5, column=0, padx=12, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(controls, text="Blur Strength", font=ctk.CTkFont(weight="bold")).grid(
            row=6, column=0, padx=12, pady=(8, 2), sticky="w"
        )
        ctk.CTkLabel(
            controls,
            text="Gaussian blur kernel size (odd). Bigger = more blur.",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        ).grid(row=7, column=0, padx=12, pady=(0, 6), sticky="w")

        self.blur_slider = ctk.CTkSlider(controls, from_=15, to=121, number_of_steps=106, command=self.on_blur_change)
        self.blur_slider.set(self.blur_strength.get())
        self.blur_slider.grid(row=8, column=0, padx=12, pady=(0, 4), sticky="ew")

        self.blur_value_lbl = ctk.CTkLabel(controls, text=f"{self.blur_strength.get()}")
        self.blur_value_lbl.grid(row=9, column=0, padx=12, pady=(0, 10), sticky="w")

        self.half_chk = ctk.CTkCheckBox(controls, text="Half precision (GPU speed-up)", variable=self.use_half)
        self.half_chk.grid(row=10, column=0, padx=12, pady=(6, 2), sticky="w")
        ctk.CTkLabel(
            controls,
            text="Enable only if CUDA GPU is available. Improves speed.",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        ).grid(row=11, column=0, padx=12, pady=(0, 8), sticky="w")

        self.preview_chk = ctk.CTkCheckBox(controls, text="Show preview window", variable=self.preview)
        self.preview_chk.grid(row=12, column=0, padx=12, pady=(2, 2), sticky="w")
        ctk.CTkLabel(
            controls,
            text="Shows live output. Press 'q' to stop from the preview window.",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        ).grid(row=13, column=0, padx=12, pady=(0, 10), sticky="w")

        btn_row = ctk.CTkFrame(controls, fg_color="transparent")
        btn_row.grid(row=14, column=0, padx=12, pady=(6, 12), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        self.run_btn = ctk.CTkButton(btn_row, text="Run", command=self.on_run)
        self.run_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_row, text="Stop", fg_color="#8b1d1d", hover_color="#a12525",
            command=self.on_stop, state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        # -----------------------------
        # Progress panel
        # -----------------------------
        prog = ctk.CTkFrame(right, corner_radius=12)
        prog.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        prog.grid_columnconfigure(0, weight=1)
        prog.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(prog, text="Progress", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 6), sticky="w"
        )
        ctk.CTkLabel(
            prog,
            text="Displays completion percentage and estimated remaining time (ETA).",
            font=ctk.CTkFont(size=12),
            text_color="#A9A9A9"
        ).grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(prog)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=2, column=0, padx=12, pady=(0, 6), sticky="ew")

        self.progress_lbl = ctk.CTkLabel(prog, text="0%  |  ETA: --:--")
        self.progress_lbl.grid(row=3, column=0, padx=12, pady=(0, 12), sticky="w")

    def _slider_row(self, parent, label, helper, var, from_, to, row):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=0, padx=12, pady=(8, 0), sticky="ew")
        wrap.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(wrap, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(top, text=label, font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        value_lbl = ctk.CTkLabel(top, text=f"{var.get():.2f}")
        value_lbl.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(wrap, text=helper, font=ctk.CTkFont(size=12), text_color="#A9A9A9").grid(
            row=1, column=0, pady=(0, 6), sticky="w"
        )

        slider = ctk.CTkSlider(
            wrap,
            from_=from_,
            to=to,
            number_of_steps=200,
            command=lambda v: (var.set(float(v)), value_lbl.configure(text=f"{float(v):.2f}"))
        )
        slider.set(var.get())
        slider.grid(row=2, column=0, sticky="ew")

    # -----------------------------
    # UI actions
    # -----------------------------
    def on_drop(self, event):
        path = event.data.strip()
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]

        # If multiple files were dropped, keep the first one
        if " " in path:
            first = path.split(" ")[0]
            if os.path.exists(first):
                path = first

        if os.path.exists(path):
            self.video_path.set(path)
        else:
            messagebox.showerror("Drop Error", "Could not read dropped file path.")

    def browse_video(self):
        fp = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )
        if fp:
            self.video_path.set(fp)

    def browse_model(self):
        fp = filedialog.askopenfilename(
            title="Select model (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")]
        )
        if fp:
            self.model_path.set(fp)

    def save_as(self):
        fp = filedialog.asksaveasfilename(
            title="Save output as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")]
        )
        if fp:
            self.output_path.set(fp)

    def on_imgsz_change(self, v):
        try:
            self.imgsz.set(int(v))
        except Exception:
            pass

    def on_blur_change(self, v):
        val = int(round(float(v)))
        if val % 2 == 0:
            val += 1
        self.blur_strength.set(val)
        self.blur_value_lbl.configure(text=str(val))

    # -----------------------------
    # Run / Stop
    # -----------------------------
    def on_run(self):
        if self.running:
            return

        vid = self.video_path.get().strip()
        mdl = self.model_path.get().strip()
        out = self.output_path.get().strip()

        if not vid or not os.path.exists(vid):
            messagebox.showerror("Error", "Please select a valid video file.")
            return
        if not mdl or not os.path.exists(mdl):
            messagebox.showerror("Error", "Please select a valid model (.pt) file.")
            return
        if not out:
            messagebox.showerror("Error", "Please choose an output path.")
            return

        self.running = True
        self.stop_flag = False
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0.0)
        self.progress_lbl.configure(text="0%  |  ETA: --:--")

        threading.Thread(target=self.worker, daemon=True).start()

    def on_stop(self):
        if self.running:
            self.stop_flag = True
            self.stop_btn.configure(state="disabled")

    # -----------------------------
    # Processing worker
    # -----------------------------
    def worker(self):
        try:
            vid = self.video_path.get().strip()
            mdl = self.model_path.get().strip()
            out = self.output_path.get().strip()

            conf = float(self.conf.get())
            iou = float(self.iou.get())
            imgsz = int(self.imgsz.get())
            blur_strength = int(self.blur_strength.get())
            show_preview = bool(self.preview.get())

            device = "cuda" if torch.cuda.is_available() else "cpu"
            half = bool(self.use_half.get()) and (device == "cuda")

            model = YOLO(mdl).to(device)

            # Read video metadata for progress calculation
            cap = cv2.VideoCapture(vid)
            if not cap.isOpened():
                raise RuntimeError("Could not open input video.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if fps <= 0:
                fps = 25

            # Open output writer
            ext = os.path.splitext(out)[1].lower()
            if ext == ".avi":
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(out, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Could not open output writer. Try another path or .avi output.")

            # Blur kernel must be odd
            k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            ksize = (k, k)

            processed = 0
            t0 = time.time()

            # Stream inference from Ultralytics
            for res in model.predict(
                source=vid,
                stream=True,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=0 if device == "cuda" else "cpu",
                half=half,
                verbose=False
            ):
                if self.stop_flag:
                    break

                frame = res.orig_img
                boxes = res.boxes

                if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy
                    if hasattr(xyxy, "cpu"):
                        xyxy = xyxy.cpu()
                    xyxy = xyxy.numpy().astype(int)

                    for (x1, y1, x2, y2) in xyxy:
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(0, min(x2, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(0, min(y2, h - 1))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, ksize, 0)

                writer.write(frame)
                processed += 1

                # Update progress and ETA
                elapsed = time.time() - t0
                fps_proc = processed / elapsed if elapsed > 0 else 0.0

                if total_frames > 0:
                    p = processed / total_frames
                    remaining = (total_frames - processed) / fps_proc if fps_proc > 0 else -1
                else:
                    p = 0.0
                    remaining = -1

                self.after(0, self.update_progress, p, remaining)

                # Optional live preview
                if show_preview:
                    cv2.imshow("Faces Blurred (Preview)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_flag = True
                        break

            writer.release()
            cv2.destroyAllWindows()

            if self.stop_flag:
                self.after(0, lambda: messagebox.showinfo("Stopped", "Processing stopped by user."))
            else:
                self.after(0, lambda: messagebox.showinfo("Done", f"Saved:\n{out}"))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.running = False
            self.stop_flag = False
            self.after(0, self.reset_buttons)

    def update_progress(self, p: float, remaining: float):
        p = max(0.0, min(1.0, float(p)))
        self.progress_bar.set(p)
        percent = int(p * 100)
        eta_txt = fmt_time(remaining) if remaining >= 0 else "--:--"
        self.progress_lbl.configure(text=f"{percent}%  |  ETA: {eta_txt}")

    def reset_buttons(self):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")


if __name__ == "__main__":
    app = App()
    app.mainloop()