# Index
1. Week
   - Homogenous/inhomogenous points
   - Pinhole model
   - Camera metrix
   - Rigid transformation
   - Projection matrix
2. Week
3. Week
4. Week
5. Week
6. Week
7. Week
8. Week
9. Week
0. Week
1. Week
2. Week
3. Week


# Week 1
**Inhomogenous coordinates** "regular coordinates".  
**Homogeneous coordinates** concats a new constant dim 1 for ease of math.

When projecting from a higher dim space into a lower one, the higher dim inhom coord, can be treated as a lower dim homo point.

**Lines**
$$
0 = x + 2y - 3 = 
\left[
\begin{array}{ccc}
1 \\
2 \\
-3
\end{array}
\right] ^ T
\left[
\begin{array}{ccc}
sx \\
sy \\
s
\end{array}
\right]$$


**Camera matrix**. Takes a homogenous point and projects it into the image plane (pixel sensors in camera).

$$\mathbf{p}_h = \mathbf{KP}$$

$$
\mathbf{K} = \left[
\begin{array}{ccc}
f & 0 & \delta_x \\
0 & f & \delta_y \\
0 & 0 & 1
\end{array}
\right]
$$

Here $\delta$ is the principle point used to translate $(0,0)$ to the upper left corner, and $f$ is the focal length, which is the distance between the principle point and the image plane. Typically half the resolution of the camera.

The camera matrix is intrensic, it does not change when the camera moves.

**Rigid transformation** allows for convertion between between coordinates systems. This is extrinsic as it changes when the camera moves.

**Projection matrix** is the product of the camera matrix and the rigid transformations.


