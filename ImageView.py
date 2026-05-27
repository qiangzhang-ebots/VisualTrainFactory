from __future__ import annotations

import sys


def _import_qt_widgets():
	if 'PySide6' in sys.modules:
		from PySide6.QtCore import QPointF, Qt
		from PySide6.QtGui import QImage, QPainter, QPixmap
		from PySide6.QtWidgets import QWidget

		return QWidget, QImage, QPainter, QPixmap, QPointF, Qt

	if 'PyQt5' in sys.modules:
		from PyQt5.QtCore import QPointF, Qt
		from PyQt5.QtGui import QImage, QPainter, QPixmap
		from PyQt5.QtWidgets import QWidget

		return QWidget, QImage, QPainter, QPixmap, QPointF, Qt

	if 'PySide2' in sys.modules:
		from PySide2.QtCore import QPointF, Qt
		from PySide2.QtGui import QImage, QPainter, QPixmap
		from PySide2.QtWidgets import QWidget

		return QWidget, QImage, QPainter, QPixmap, QPointF, Qt

	try:
		from PySide6.QtCore import QPointF, Qt
		from PySide6.QtGui import QImage, QPainter, QPixmap
		from PySide6.QtWidgets import QWidget

		return QWidget, QImage, QPainter, QPixmap, QPointF, Qt
	except ImportError:
		pass

	try:
		from PyQt5.QtCore import QPointF, Qt
		from PyQt5.QtGui import QImage, QPainter, QPixmap
		from PyQt5.QtWidgets import QWidget

		return QWidget, QImage, QPainter, QPixmap, QPointF, Qt
	except ImportError:
		pass

	try:
		from PySide2.QtCore import QPointF, Qt
		from PySide2.QtGui import QImage, QPainter, QPixmap
		from PySide2.QtWidgets import QWidget

		return QWidget, QImage, QPainter, QPixmap, QPointF, Qt
	except ImportError as exc:
		raise ImportError('Need PySide6, PyQt5, or PySide2 installed') from exc


QWidget, QImage, QPainter, QPixmap, QPointF, Qt = _import_qt_widgets()


try:
	import cv2
except ImportError as exc:
	raise ImportError('Need opencv-python installed') from exc


class ImageView(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setMinimumSize(320, 240)
		self.setMouseTracking(True)
		self._pixmap = QPixmap()
		self._scale = 1.0
		self._offset = QPointF(0.0, 0.0)
		self._dragging = False
		self._drag_start_pos = None
		self._base_offset = QPointF(0.0, 0.0)
		self._auto_fit = True

	def sizeHint(self):
		return self.minimumSize()

	def SetImage(self, image):
		self._pixmap = self._to_pixmap(image)
		self._scale = 1.0
		self._offset = QPointF(0.0, 0.0)
		self._dragging = False
		self._drag_start_pos = None
		self._auto_fit = True
		self._fit_to_widget()
		self.update()

	def clear(self):
		self._pixmap = QPixmap()
		self._scale = 1.0
		self._offset = QPointF(0.0, 0.0)
		self._auto_fit = True
		self.update()

	def _fit_to_widget(self):
		if self._pixmap.isNull():
			return

		view_width = max(1, self.width())
		view_height = max(1, self.height())
		image_width = max(1, self._pixmap.width())
		image_height = max(1, self._pixmap.height())

		self._scale = min(view_width / image_width, view_height / image_height)
		self._offset = QPointF(
			(view_width - image_width * self._scale) / 2.0,
			(view_height - image_height * self._scale) / 2.0,
		)

	def _to_pixmap(self, image):
		if image is None:
			return QPixmap()

		if isinstance(image, QPixmap):
			return image.copy()

		if isinstance(image, str):
			loaded = cv2.imread(image, cv2.IMREAD_UNCHANGED)
			if loaded is None:
				return QPixmap()
			image = loaded

		if hasattr(image, 'shape'):
			return self._cv_to_pixmap(image)

		return QPixmap()

	def _cv_to_pixmap(self, image):
		if image is None:
			return QPixmap()

		if len(image.shape) == 2:
			height, width = image.shape
			bytes_per_line = width
			qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
			return QPixmap.fromImage(qimage.copy())

		height, width, channel_count = image.shape[:3]

		if channel_count == 3:
			rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			bytes_per_line = 3 * width
			qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
			return QPixmap.fromImage(qimage.copy())

		if channel_count == 4:
			rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
			bytes_per_line = 4 * width
			qimage = QImage(rgba_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
			return QPixmap.fromImage(qimage.copy())

		return QPixmap()

	def paintEvent(self, event):
		painter = QPainter(self)
		painter.setRenderHint(QPainter.Antialiasing, True)
		painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
		painter.fillRect(self.rect(), Qt.black)

		if self._pixmap.isNull():
			return

		painter.translate(self._offset)
		painter.scale(self._scale, self._scale)
		painter.drawPixmap(0, 0, self._pixmap)

	def resizeEvent(self, event):
		super().resizeEvent(event)
		if self._pixmap.isNull():
			return

		if self._auto_fit:
			self._fit_to_widget()
			self.update()

	def wheelEvent(self, event):
		if self._pixmap.isNull():
			return

		delta = event.angleDelta().y()
		if delta == 0:
			return

		cursor_pos = event.position() if hasattr(event, 'position') else event.posF()
		if cursor_pos is None:
			cursor_pos = QPointF(self.width() / 2.0, self.height() / 2.0)

		self._auto_fit = False

		old_scale = self._scale
		zoom_factor = 1.15 if delta > 0 else 1 / 1.15
		new_scale = max(0.05, min(old_scale * zoom_factor, 40.0))

		if old_scale <= 0:
			old_scale = 1.0

		image_point_x = (cursor_pos.x() - self._offset.x()) / old_scale
		image_point_y = (cursor_pos.y() - self._offset.y()) / old_scale

		self._scale = new_scale
		self._offset = QPointF(
			cursor_pos.x() - image_point_x * new_scale,
			cursor_pos.y() - image_point_y * new_scale,
		)
		self.update()

	def mousePressEvent(self, event):
		if self._pixmap.isNull() or event.button() != Qt.LeftButton:
			return

		self._auto_fit = False
		self._dragging = True
		self._drag_start_pos = event.pos()
		self._base_offset = QPointF(self._offset)
		self.setCursor(Qt.ClosedHandCursor)

	def mouseMoveEvent(self, event):
		if not self._dragging or self._drag_start_pos is None:
			return

		delta = event.pos() - self._drag_start_pos
		self._offset = QPointF(self._base_offset.x() + delta.x(), self._base_offset.y() + delta.y())
		self.update()

	def mouseReleaseEvent(self, event):
		if event.button() != Qt.LeftButton:
			return

		self._dragging = False
		self._drag_start_pos = None
		self.unsetCursor()

	def leaveEvent(self, event):
		if not self._dragging:
			self.unsetCursor()

