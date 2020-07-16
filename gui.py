import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QTextEdit, QPushButton, QLabel, QVBoxLayout)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir

filenames = []

class DialogApp(QWidget):
	def __init__(self):
		super().__init__()
		self.resize(800, 600)

		self.button2 = QPushButton('Import Python Script')
		self.button2.clicked.connect(self.get_text_file)

		layout = QVBoxLayout()
		layout.addWidget(self.button2)
		self.setLayout(layout)

	def get_text_file(self):
		dialog = QFileDialog()
		dialog.setFileMode(QFileDialog.AnyFile)
		dialog.setFilter(QDir.Files)
		self.filenames = []
		if dialog.exec_():
			self.filenames.append(dialog.selectedFiles()[0])
			print(self.filenames)

if __name__ == '__main__':
	app = QApplication(sys.argv)

	demo = DialogApp()
	demo.show()
	sys.exit(app.exec_())
