# Sudoku solver

Solver takes in image of nxn sudoku grid, preprocesses image, and crops image to image of grid. From there, the grid is split into nxn images, where each image corresponds to a square. A simple SVC is used to detect and return digits in each square. The puzzle is then solved using a recursive backtracing algorithm. The final result is an overlay of the solution onto the cropped grid.

Before: ![starting_grid](https://user-images.githubusercontent.com/17757035/46920421-ffea5900-cfa2-11e8-9b90-18e4df28690f.jpg)

After: ![solved_grid](https://user-images.githubusercontent.com/17757035/46920422-07a9fd80-cfa3-11e8-8ae0-ad96acbbee72.png)

Issues: 
1. Image quality deteriorates after various processing and resizing operations. As of now, unable to fix this without affecting consistency of process.
2. If SVC is unable to perfectly detect the digits in the starting grid, the final solution will be off. If this happens, retraining the SVC using the provided functions in svm.py will be necessary
3. Solving an nxn sudoku grid is known to be NP-complete. Thus, for any n>9, solving the grid using this project will likely be infeasible. 

Next steps: fix image quality deterioration, work on functionality for solving grids seen in video.
