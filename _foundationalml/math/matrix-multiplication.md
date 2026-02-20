---
title: 'Matrix Multiplcation'
permalink: /foundational-ml/math/matrix-multiplication
tags:
  - machine learning
  - foundational-ml
  - math
comments: true
description: What are Matrices? How do you Multiply them? Why do you need them in ML?
---

## What are Matrices?

Matrices are a rectangular arrangement of data. They can be numbers, variables, symbols or expressions. These are represented in rows and columns.
Here's an example:
$$
numbers = \begin{bmatrix}
1 & 4 & 7 \\
2 & 5 & 8 \\
3 & 6 & 9
\end{bmatrix}
$$
or
$$
fruits = \begin{bmatrix}
banana & apple & mango \\
jackfruit & tomato & papaya \\
\end{bmatrix}
$$

The shape of a matrix is represented as the `number of rows X number of columns`. The shape of the matrix `numbers` is `3X3` and the shape of the matrix fruits is `2X3`.

## Matrix Multiplication
Consider two matrices
$$
A = \begin{bmatrix}
a_{00} & a_{01} & a_{02} \\
a_{10} & a_{11} & a_{12} \\
a_{20} & a_{21} & a_{22}
\end{bmatrix}
,
B = \begin{bmatrix}
b_{00} & b_{01} & b_{02} \\
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
$$

To multiply two matrices, you the the first row of the first matrix, and do a scalar multiplcation with the first column of the second matrix. This is the value of the first row's first column. Repeat the process to build out the entire result matrix.

The product of these matrices $$A \cdot B$$ or just $$AB$$ is as follows.
$$
B = \begin{bmatrix}
a_{00}b_{00} + a_{01}b_{10} + a_{02}b_{20} & a_{00}b_{01} + a_{01}b_{11} + a_{02}b_{21} & a_{00}b_{02} + a_{01}b_{12} + a_{02}b_{22} \\
a_{10}b_{00} + a_{11}b_{10} + a_{12}b_{20} & a_{10}b_{01} + a_{11}b_{11} + a_{12}b_{21} & a_{10}b_{02} + a_{11}b_{12} + a_{12}b_{22} \\
a_{20}b_{00} + a_{21}b_{10} + a_{22}b_{20} & a_{20}b_{01} + a_{21}b_{11} + a_{22}b_{21} & a_{20}b_{02} + a_{21}b_{12} + a_{22}b_{22} 
\end{bmatrix}
$$

Check [this](https://www.mathsisfun.com/algebra/matrix-multiplying.html) out if you want a more visual explanation or just scroll to the end of this page for a visualizer.

* Matrix multiplication is not Commutative. So $$AB != BA$$.
* To be able to multiply two matrices, the number of rows of the first matrix should be equal to the number of rows of the second matrix.

## Why is this important in ML?
Machine Learning has a lot of linear algebra. We can represent these operations as matrices.

For example, consider the linear equation $$y = ax_{0} + bx_{1} + cx_{2}$$. We can represent this using matrices as
$$
y = \begin{bmatrix}
a & b & c
\end{bmatrix} \times \begin{bmatrix}
x_0\\
x_1\\
x_2
\end{bmatrix}
$$ 

Another advantage of using Matrices in machine learning is that it lets us speed up our processing. GPUs are really poweful at executing instructions in parallel that you can parallelize a large number of operations if you can represent them as matrix operations and perform them on a GPU.

<div class="simd-visualizer" style="font-family: system-ui, sans-serif; max-width: 600px; margin: 0 auto; text-align: center; border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: #fafafa;">
  <h3 style="margin-top: 0;">Matrix Multiplication: CPU vs GPU</h3>
  
  <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 20px;">
    <div id="matA" style="display: grid; grid-template-columns: repeat(3, 30px); gap: 5px;"></div>
    <div style="font-weight: bold; color: #555;">X</div>
    <div id="matB" style="display: grid; grid-template-columns: repeat(3, 30px); gap: 5px;"></div>
    <div style="font-weight: bold; color: #555;">=</div>
    <div id="matC" style="display: grid; grid-template-columns: repeat(3, 30px); gap: 5px;"></div>
  </div>

  <div style="display: flex; justify-content: center; gap: 10px;">
    <button onclick="runCPU()" style="padding: 10px 15px; border: none; background: #007bff; color: white; border-radius: 5px; cursor: pointer; font-weight: bold;">Compute Sequentially (CPU)</button>
    <button onclick="runGPU()" style="padding: 10px 15px; border: none; background: #28a745; color: white; border-radius: 5px; cursor: pointer; font-weight: bold;">Compute in Parallel (GPU)</button>
    <button onclick="resetVis()" style="padding: 10px 15px; border: 1px solid #ccc; background: #fff; border-radius: 5px; cursor: pointer;">Reset</button>
  </div>

  <p id="vis-status" style="margin-top: 15px; font-weight: bold; min-height: 20px; color: #333;"></p>

  <style>
    .cell { width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; background: #fff; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; transition: all 0.2s; }
    .cell.active-a { background: #ffeeba; border-color: #ffc107; transform: scale(1.1); z-index: 10;}
    .cell.active-b { background: #b8daff; border-color: #007bff; transform: scale(1.1); z-index: 10;}
    .cell.result-active { background: #c3e6cb; border-color: #28a745; font-weight: bold; }
  </style>

  <script>
    const A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    const B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]];
    let animationTimeout;

    function renderMatrix(id, data, empty = false) {
      const container = document.getElementById(id);
      container.innerHTML = '';
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const cell = document.createElement('div');
          cell.className = 'cell';
          cell.id = `${id}-${i}-${j}`;
          cell.innerText = empty ? '' : data[i][j];
          container.appendChild(cell);
        }
      }
    }

    function resetVis() {
      clearTimeout(animationTimeout);
      renderMatrix('matA', A);
      renderMatrix('matB', B);
      renderMatrix('matC', [], true);
      document.getElementById('vis-status').innerText = 'Ready.';
    }

    function computeCell(r, c) {
      let sum = 0;
      for (let k = 0; k < 3; k++) sum += A[r][k] * B[k][c];
      return sum;
    }

    function clearHighlights() {
      document.querySelectorAll('.cell').forEach(el => el.classList.remove('active-a', 'active-b', 'result-active'));
    }

    function highlightRowCol(r, c) {
      clearHighlights();
      for(let k=0; k<3; k++) {
        document.getElementById(`matA-${r}-${k}`).classList.add('active-a');
        document.getElementById(`matB-${k}-${c}`).classList.add('active-b');
      }
    }

    function runCPU() {
      resetVis();
      document.getElementById('vis-status').innerText = 'Computing sequentially...';
      let r = 0, c = 0;

      function step() {
        if (r >= 3) {
          clearHighlights();
          document.getElementById('vis-status').innerText = 'Done! 9 iterations completed.';
          return;
        }
        highlightRowCol(r, c);
        const resCell = document.getElementById(`matC-${r}-${c}`);
        resCell.innerText = computeCell(r, c);
        resCell.classList.add('result-active');

        c++;
        if (c >= 3) { c = 0; r++; }
        animationTimeout = setTimeout(step, 600);
      }
      step();
    }

    function runGPU() {
      resetVis();
      document.getElementById('vis-status').innerText = 'Dispatching threads... Computing in parallel!';
      
      // Highlight everything
      for(let i=0; i<3; i++) {
        for(let j=0; j<3; j++) {
          document.getElementById(`matA-${i}-${j}`).classList.add('active-a');
          document.getElementById(`matB-${i}-${j}`).classList.add('active-b');
        }
      }

      // Compute all instantly
      animationTimeout = setTimeout(() => {
        for(let r=0; r<3; r++) {
          for(let c=0; c<3; c++) {
            const resCell = document.getElementById(`matC-${r}-${c}`);
            resCell.innerText = computeCell(r, c);
            resCell.classList.add('result-active');
          }
        }
        document.getElementById('vis-status').innerText = 'Done! 1 step completed across 9 parallel threads.';
      }, 500); // Slight delay for dramatic effect
    }

    // Initialize
    resetVis();
  </script>
</div>

There are more uses of matrices in machine learning. You'll learn about them as you understand the algorithms and their implementations.