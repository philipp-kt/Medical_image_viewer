"""
DICOM Medical Image Viewer
A simple GUI application for browsing and displaying DICOM images.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import pydicom

try:
    from config import DEFAULT_DICOM_DIR
except ImportError:
    # No local config.py present — see config.example.py to set a default
    # folder to auto-load on startup. Otherwise use File > Open Folder.
    DEFAULT_DICOM_DIR = ""


class DicomViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DICOM Viewer")
        self.geometry("1100x700")
        self.minsize(800, 500)

        self.current_dataset = None
        self.photo_image = None  # prevent garbage collection
        self.window_center = None
        self.window_width = None
        self.zoom_level = 1.0
        self.scale_fit = 1.0  # fit-to-canvas scale factor
        self.pan_offset = [0, 0]  # [x, y] offset for panning
        self._pan_start = None

        self._build_menu()
        self._build_ui()

        # Auto-load default directory if it exists
        if os.path.isdir(DEFAULT_DICOM_DIR):
            self._populate_tree(DEFAULT_DICOM_DIR)

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Folder…", command=self._open_folder)
        file_menu.add_command(label="Export Image…", command=self._export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_ui(self):
        # Main paned window: left tree | right image + info
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # --- Left: file tree ---
        left_frame = ttk.Frame(paned, width=300)
        paned.add(left_frame, weight=1)

        ttk.Label(left_frame, text="DICOM Files", font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W, padx=4, pady=(4, 0)
        )

        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.tree = ttk.Treeview(tree_frame, show="tree headings")
        self.tree.heading("#0", text="Name", anchor=tk.W)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # --- Right: image + metadata ---
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        # Image canvas
        self.canvas = tk.Canvas(right_frame, bg="#1e1e1e")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)

        # Zoom controls
        zoom_frame = ttk.Frame(right_frame)
        zoom_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.zoom_label = ttk.Label(zoom_frame, text="Zoom: —", font=("Segoe UI", 9))
        self.zoom_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(zoom_frame, text="Fit", width=4, command=self._reset_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="1:1", width=3, command=self._zoom_native).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="+", width=2, command=lambda: self._zoom(1.25)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="\u2013", width=2, command=lambda: self._zoom(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Label(zoom_frame, text="Scroll to zoom, drag to pan", foreground="gray",
                  font=("Segoe UI", 8)).pack(side=tk.RIGHT, padx=4)

        # Window / Level controls
        controls_frame = ttk.LabelFrame(right_frame, text="Window / Level")
        controls_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        ttk.Label(controls_frame, text="Center:").grid(row=0, column=0, padx=4, pady=2)
        self.wc_var = tk.DoubleVar(value=0)
        self.wc_slider = ttk.Scale(
            controls_frame, from_=-1024, to=3096, variable=self.wc_var,
            orient=tk.HORIZONTAL, command=self._on_wl_change,
        )
        self.wc_slider.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        self.wc_label = ttk.Label(controls_frame, text="0", width=6)
        self.wc_label.grid(row=0, column=2, padx=4)

        ttk.Label(controls_frame, text="Width:").grid(row=1, column=0, padx=4, pady=2)
        self.ww_var = tk.DoubleVar(value=1)
        self.ww_slider = ttk.Scale(
            controls_frame, from_=1, to=4096, variable=self.ww_var,
            orient=tk.HORIZONTAL, command=self._on_wl_change,
        )
        self.ww_slider.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        self.ww_label = ttk.Label(controls_frame, text="1", width=6)
        self.ww_label.grid(row=1, column=2, padx=4)

        controls_frame.columnconfigure(1, weight=1)

        ttk.Button(controls_frame, text="Reset W/L", command=self._reset_wl).grid(
            row=0, column=3, rowspan=2, padx=8, pady=2
        )

        # Metadata panel
        meta_frame = ttk.LabelFrame(right_frame, text="DICOM Metadata")
        meta_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.meta_text = tk.Text(meta_frame, height=6, wrap=tk.WORD, font=("Consolas", 9))
        meta_scroll = ttk.Scrollbar(meta_frame, orient=tk.VERTICAL, command=self.meta_text.yview)
        self.meta_text.configure(yscrollcommand=meta_scroll.set, state=tk.DISABLED)
        self.meta_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        meta_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # --------------------------------------------------------- File tree
    def _open_folder(self):
        folder = filedialog.askdirectory(title="Select DICOM folder")
        if folder:
            self._populate_tree(folder)

    def _export_image(self):
        if not hasattr(self, "pixel_array"):
            messagebox.showinfo("Info", "No image loaded to export.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Image",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("TIFF", "*.tiff *.tif"),
                ("BMP", "*.bmp"),
            ],
        )
        if not filepath:
            return

        # Apply current window/level to get the displayed image
        center = self.wc_var.get()
        width = self.ww_var.get()
        img_array = self._apply_window_level(self.pixel_array, center, width)

        if img_array.ndim == 3 and img_array.shape[2] in (3, 4):
            pil_img = Image.fromarray(img_array)
        else:
            if img_array.ndim == 3:
                img_array = img_array[0]
            pil_img = Image.fromarray(img_array, mode="L")

        try:
            pil_img.save(filepath)
            messagebox.showinfo("Export", f"Image saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def _populate_tree(self, root_dir):
        # Clear existing tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        root_path = Path(root_dir)
        self.tree.insert("", tk.END, iid=str(root_path), text=root_path.name, open=True)
        self._add_tree_children(str(root_path), root_path)

    def _add_tree_children(self, parent_iid, parent_path):
        try:
            entries = sorted(parent_path.iterdir())
        except PermissionError:
            return

        for entry in entries:
            iid = str(entry)
            if entry.is_dir():
                self.tree.insert(parent_iid, tk.END, iid=iid, text=entry.name)
                self._add_tree_children(iid, entry)
            elif entry.is_file():
                # Show all files – DICOM files often have no extension
                self.tree.insert(parent_iid, tk.END, iid=iid, text=entry.name)

    def _on_tree_select(self, _event):
        selection = self.tree.selection()
        if not selection:
            return
        path = Path(selection[0])
        if path.is_file():
            self._load_dicom(path)

    # -------------------------------------------------------- DICOM load
    def _load_dicom(self, filepath):
        try:
            ds = pydicom.dcmread(str(filepath), force=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read DICOM file:\n{e}")
            return

        if not hasattr(ds, "pixel_array"):
            try:
                _ = ds.PixelData
            except AttributeError:
                messagebox.showinfo("Info", "Selected file has no image data.")
                self._show_metadata(ds)
                return

        try:
            pixel_array = ds.pixel_array.astype(np.float64)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot decode pixel data:\n{e}")
            self._show_metadata(ds)
            return

        # Apply rescale slope / intercept if present
        slope = getattr(ds, "RescaleSlope", 1)
        intercept = getattr(ds, "RescaleIntercept", 0)
        pixel_array = pixel_array * float(slope) + float(intercept)

        self.current_dataset = ds
        self.pixel_array = pixel_array
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]

        # Set initial window/level from DICOM tags or data range
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            # Can be multi-valued
            self.window_center = float(wc[0]) if isinstance(wc, pydicom.multival.MultiValue) else float(wc)
            self.window_width = float(ww[0]) if isinstance(ww, pydicom.multival.MultiValue) else float(ww)
        else:
            self.window_center = float(np.mean(pixel_array))
            self.window_width = float(np.ptp(pixel_array))

        # Update slider ranges based on data
        data_min, data_max = float(np.min(pixel_array)), float(np.max(pixel_array))
        margin = max(abs(data_max - data_min) * 0.5, 100)
        self.wc_slider.configure(from_=data_min - margin, to=data_max + margin)
        self.ww_slider.configure(from_=1, to=(data_max - data_min) * 2 + 1)

        self.wc_var.set(self.window_center)
        self.ww_var.set(self.window_width)

        self._update_image()
        self._show_metadata(ds)

    # ------------------------------------------------------- Image display
    def _apply_window_level(self, pixel_array, center, width):
        lower = center - width / 2
        upper = center + width / 2
        img = np.clip(pixel_array, lower, upper)
        # Normalize to 0-255
        if upper != lower:
            img = (img - lower) / (upper - lower) * 255.0
        else:
            img = np.zeros_like(pixel_array)
        return img.astype(np.uint8)

    def _update_image(self):
        if not hasattr(self, "pixel_array"):
            return

        center = self.wc_var.get()
        width = self.ww_var.get()
        img_array = self._apply_window_level(self.pixel_array, center, width)

        # Handle multi-frame or RGB
        if img_array.ndim == 3 and img_array.shape[2] in (3, 4):
            pil_img = Image.fromarray(img_array)
        else:
            # If 3D (multi-frame), take the first frame
            if img_array.ndim == 3:
                img_array = img_array[0]
            pil_img = Image.fromarray(img_array, mode="L")

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        # Compute base size that fits the canvas (zoom_level 1.0 = fit)
        img_w, img_h = pil_img.size
        self.scale_fit = min(cw / img_w, ch / img_h)
        display_scale = self.scale_fit * self.zoom_level
        new_w = max(1, int(img_w * display_scale))
        new_h = max(1, int(img_h * display_scale))
        self.zoom_label.config(text=f"Zoom: {display_scale * 100:.0f}%")
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        x = cw // 2 + self.pan_offset[0]
        y = ch // 2 + self.pan_offset[1]
        self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.photo_image)

    def _on_canvas_resize(self, _event):
        if hasattr(self, "pixel_array"):
            self._update_image()

    # ------------------------------------------------------------ Zoom / Pan
    def _zoom(self, factor):
        new_zoom = self.zoom_level * factor
        new_zoom = max(0.1, min(new_zoom, 20.0))
        # Scale pan offset so the center stays consistent
        if self.zoom_level > 0:
            ratio = new_zoom / self.zoom_level
            self.pan_offset[0] = int(self.pan_offset[0] * ratio)
            self.pan_offset[1] = int(self.pan_offset[1] * ratio)
        self.zoom_level = new_zoom
        self._update_image()

    def _zoom_native(self):
        """Set zoom so image displays at native (1:1) pixel size."""
        if self.scale_fit > 0:
            self.zoom_level = 1.0 / self.scale_fit
            self.pan_offset = [0, 0]
            self._update_image()

    def _reset_zoom(self):
        """Fit image to canvas."""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self._update_image()

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self._zoom(1.15)
        else:
            self._zoom(1 / 1.15)

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self.pan_offset[0] += dx
        self.pan_offset[1] += dy
        self._update_image()

    def _on_pan_end(self, _event):
        self._pan_start = None

    def _on_wl_change(self, _value=None):
        self.wc_label.config(text=f"{self.wc_var.get():.0f}")
        self.ww_label.config(text=f"{self.ww_var.get():.0f}")
        self._update_image()

    def _reset_wl(self):
        if self.window_center is not None:
            self.wc_var.set(self.window_center)
            self.ww_var.set(self.window_width)
            self._on_wl_change()

    # ----------------------------------------------------------- Metadata
    def _show_metadata(self, ds):
        self.meta_text.configure(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)

        fields = [
            ("Patient Name", "PatientName"),
            ("Patient ID", "PatientID"),
            ("Study Date", "StudyDate"),
            ("Modality", "Modality"),
            ("Study Description", "StudyDescription"),
            ("Series Description", "SeriesDescription"),
            ("Image Size", None),
            ("Bits Stored", "BitsStored"),
            ("Photometric", "PhotometricInterpretation"),
        ]

        for label, attr in fields:
            if attr is None and hasattr(ds, "Rows"):
                val = f"{ds.Rows} x {ds.Columns}"
            elif attr:
                val = getattr(ds, attr, "N/A")
            else:
                val = "N/A"
            self.meta_text.insert(tk.END, f"{label}: {val}\n")

        self.meta_text.configure(state=tk.DISABLED)


if __name__ == "__main__":
    app = DicomViewer()
    app.mainloop()
