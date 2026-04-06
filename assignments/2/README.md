# Assignment 2
*This project has been created as a part of the Neural Networks course at UCT Prague by Vladimir Nikolaev*
## Initial Plan
The goal of this assignment is to build an interactive app that uses PyTorch for deep learning and PyQt5 for the graphical user interface (GUI).
The plan is to:
1. Create a custom drawing area (canvas) where the user can click to add points.
2. Create a PyTorch model (`nn.Module`) that automatically builds its hidden layers based on the numbers typed by the user.
3. Show the neural network's predictions for every pixel on the canvas, creating a colored map that shows how the network separates the red and blue points.

## Neural Networks

- **Inputs**: An `(x, y)` position of each pixel on the canvas as input data.
- **Hidden Layers**: The network uses linear layers (`nn.Linear`) and non-linear functions (`nn.ReLU`) to bend and warp the space. This is important because it allows the network to draw curved shapes to group the points, instead of just straight lines.
- **Outputs & Loss**: The final layer has 2 outputs (for the RED and BLUE classes). The error is calculated (through `CrossEntropyLoss`) between the network's guess and the actual color of the point the user clicked.
- **Training**: The `Adam` optimizer is used to slightly adjust the network's settings over many steps (epochs) to reduce its mistakes.
- **Showing the Results**: To see what the network has learned, it is tested on every pixel on the canvas. The network gives a probability (through `Softmax`) of whether a pixel should be RED or BLUE. These probabilities are used to set the Red and Blue color of each pixel. If the network is unsure, both red and blue are mixed, creating a purple color on the screen.

## Implementation

### `DynamicModel(nn.Module)`
- This is the PyTorch neural network model.
- Using `nn.Module` and `nn.Sequential` allows to build the network layers dynamically based on user input.

### `Canvas(QWidget)`
- The interactive drawing area.
  - `mousePressEvent` is used to detect exactly where the user clicks. A left-click adds a RED point, and a right-click adds a BLUE point.
  - `paintEvent` is used to first paint the background colors (the network's predictions) and then draw the points the user clicked on top of it.

### `MainWindow(QMainWindow)`
- The main application window.
- `QMainWindow` gives the main window with the buttons, text boxes, and the drawing canvas. It also connects the button clicks to the PyTorch module.

## Program flow

- When the user clicks "Build Network", the main window reads the text box and creates a new `DynamicModel`. 
- When the user clicks "Train Network", the canvas gives the coordinate points `(x,y)` to PyTorch.
- These points are getting normalized (between 0 and 1) so the network can learn better.
- After training, the program quickly checks every pixel on the canvas using the neural network, turns the results into an image coloring using Numpy, and tells PyQt5 to draw the new image so the user can see the updated results.

## AI Usage
AI was used to discuss the design of the main window layout, figure out how to program the interactive PyQt5 canvas, and write documentation (docstrings). NotebookLM was used to understand the neural network concepts (with uploading the presentations and examples from moodle, and various sources on the internet). However, the PyTorch neural network code, dynamic layer building, and training loops were written independently.

## Sources
- [Learn Python PyQt5 in 1 hour](https://www.youtube.com/watch?v=92zx_U9Nzf4)
- [QPainter documentation](https://doc.qt.io/qt-6/qpainter.html)
- [NumPy cheatsheet - GeeksforGeeks](https://www.geeksforgeeks.org/numpy-cheat-sheet/)
- [Learing PyTorch with Examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)
- [pytorch-examples - GitHub](https://github.com/jcjohnson/pytorch-examples)