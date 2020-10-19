import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMainWindow, QPushButton, QStyle, QLabel, QSizePolicy, QWidget, QVBoxLayout, QFileDialog, \
    QApplication


class VideoWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Display ICSI video")

        self.openButton = QPushButton("Open video")
        self.openButton.clicked.connect(self.open_video_file)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        widget = QWidget(self)
        self.setCentralWidget(widget)

        videoWidget = QVideoWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.openButton)
        layout.addWidget(self.playButton)
        layout.addWidget(self.errorLabel)
        layout.addWidget(videoWidget)

        widget.setLayout(layout)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.state_changed)
        self.mediaPlayer.error.connect(self.handle_error)

    def open_video_file(self):
        filePath, _ = QFileDialog.getOpenFileName(self, 'Choose a video file', '', 'Videos files | *.avi;')

        if filePath != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filePath)))
            self.playButton.setEnabled(True)

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def state_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def handle_error(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())