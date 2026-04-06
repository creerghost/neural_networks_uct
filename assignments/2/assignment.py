import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtGui import QPainter, QImage, QPen, QBrush
from PyQt5.QtCore import Qt


class DynamicModel(nn.Module):
    """
    A PyTorch neural network module that dynamically creates fully-connected
    hidden layers based on a list of sizes, and outputs 2 target classes
    (RED or BLUE).
    """
    def __init__(self, hidden_layers: list[int]) -> None:
        """
        Initializes the dynamic classifier with a variable number of hidden
        layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 2  # Coordinates (x, y)

        for hidden in hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden

        layers.append(nn.Linear(in_features, 2))  # Outputs: RED(0), BLUE(1)
        self.network = nn.Sequential(*layers)  # passes args instead of a list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.
        """
        return self.network(x)


class Canvas(QWidget):
    """
    A custom QWidget acting as a canvas for drawing points and visualizing the
    decision boundaries of the trained neural network model.
    """
    def __init__(self, parent=None) -> None:
        """
        Initializes the canvas, sets fixed size, and prepares an empty points
        list and background image placeholder.
        """
        super().__init__(parent)
        # List of (x, y, target_class). 0 for RED and 1 for BLUE.
        self.points: list[tuple[int, int, int]] = []
        self.setFixedSize(400, 400)
        self.bg_image: QImage | None = None

    def mousePressEvent(self, event) -> None:
        """
        Handles mouse click events on the canvas to place training points.
        Left-click places a RED point (Class 0), right-click places a BLUE
        point (Class 1).
        """
        x: int = event.x()
        y: int = event.y()

        if 0 <= x < self.width() and 0 <= y < self.height():
            if event.button() == Qt.LeftButton:
                self.points.append((x, y, 0))
            elif event.button() == Qt.RightButton:
                self.points.append((x, y, 1))
            self.update()

    def get_training_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compiles the points drawn on the canvas into PyTorch tensors for
        training.
        """
        if not self.points:
            return None, None

        X: list[list[float]] = []
        y: list[int] = []
        for px, py, color in self.points:
            # Normalize inputs into [0, 1] range based on canvas size
            X.append([px / self.width(), py / self.height()])
            y.append(color)

        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

    def update_background(self, model: nn.Module) -> None:
        """
        Evaluates the provided PyTorch model across the entire canvas area to
        generate a background image showing the decision probabilities for the
        RED and BLUE classes.
        """
        if model is None:
            return

        w, h = self.width(), self.height()

        # Create grid representing all normalized coordinates
        xs = np.linspace(0, 1, w, endpoint=False)
        ys = np.linspace(0, 1, h, endpoint=False)
        xv, yv = np.meshgrid(xs, ys)

        grid = np.c_[xv.ravel(), yv.ravel()]
        tensor_grid = torch.tensor(grid, dtype=torch.float32)

        # Batch evaluation
        batch_size = 10000
        predictions = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(tensor_grid), batch_size):
                batch = tensor_grid[i:i+batch_size]
                out = model(batch)
                prob = torch.softmax(out, dim=1)
                predictions.append(prob)

        predictions = torch.cat(predictions, dim=0).numpy()

        # Predictions to RGB map
        image_data = np.zeros((h, w, 3), dtype=np.uint8)

        probs_r = predictions[:, 0].reshape(h, w)
        probs_b = predictions[:, 1].reshape(h, w)

        image_data[:, :, 0] = np.clip(probs_r * 255, 0, 255)  # R
        image_data[:, :, 1] = 0                               # G
        image_data[:, :, 2] = np.clip(probs_b * 255, 0, 255)  # B

        qimage = QImage(image_data.data, w, h, w * 3, QImage.Format_RGB888)
        self.bg_image = qimage.copy()
        self.update()

    def paintEvent(self, event) -> None:
        """
        Paints the background image representing class probabilities and then
        overrides it with the plotted training data points.
        """
        painter = QPainter(self)

        if self.bg_image is not None:
            painter.drawImage(0, 0, self.bg_image)
        else:
            painter.fillRect(self.rect(), Qt.white)

        for x, y, color in self.points:
            color = Qt.red if color == 0 else Qt.blue
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.black, 1))
            painter.drawEllipse(x - 4, y - 4, 8, 8)


class MainWindow(QMainWindow):
    """
    Main GUI window.
    """
    def __init__(self) -> None:
        """
        Initializes the main window parameters, the PyTorch
        training components, and the UI elements.
        """
        super().__init__()
        self.setWindowTitle("NN UCT Assignment 2")

        self.model: DynamicModel | None = None
        self.optimizer: optim.Adam | None = None
        self.criterion = nn.CrossEntropyLoss()

        self.init_ui()

    def init_ui(self) -> None:
        """
        Sets up the layout, including the control panel (layer sizes,
        epoch inputs) and the interactive canvas.
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        left_panel = QVBoxLayout()

        left_panel.addWidget(QLabel("Hidden layers (comma separated)"))
        self.layers_box = QLineEdit("10, 10")
        left_panel.addWidget(self.layers_box)
        self.layers_btn = QPushButton("Build Network")
        self.layers_btn.clicked.connect(self.build_network)
        left_panel.addWidget(self.layers_btn)

        left_panel.addSpacing(20)

        left_panel.addWidget(QLabel("Number of epochs"))
        self.epochs_box = QLineEdit("100")
        left_panel.addWidget(self.epochs_box)
        self.epochs_btn = QPushButton("Train Network")
        self.epochs_btn.clicked.connect(self.train_network)
        left_panel.addWidget(self.epochs_btn)

        left_panel.addStretch()

        layout.addLayout(left_panel)

        self.canvas = Canvas()
        layout.addWidget(self.canvas)

    def build_network(self) -> None:
        """
        Parses the hidden layers text input, instantiates the
        DynamicModel, and prepares the Adam optimizer for standard
        classification.
        """
        try:
            layers_str: str = self.layers_box.text()
            if not layers_str.strip():
                hidden_layers: list[int] = []
            else:
                hidden_layers = [int(x.strip()) for x in layers_str.split(',')]

            self.model = DynamicModel(hidden_layers)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

            self.canvas.bg_image = None
            self.canvas.update()

            self.canvas.update_background(self.model)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to "
                                               f"build network:\n{e}")

    def train_network(self) -> None:
        """
        Trains the current model using the user-provided points for a specified
        number of epochs, then updates the canvas decision boundaries.
        """
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please "
                                               "construct the network first.")
            return

        try:
            epochs: int = int(self.epochs_box.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid number of epochs.")
            return

        X, y = self.canvas.get_training_data()
        if X is None:
            QMessageBox.warning(self, "Error", "Please add some points first.")
            return

        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

        self.canvas.update_background(self.model)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
