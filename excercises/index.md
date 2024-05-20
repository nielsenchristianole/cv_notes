# Index
1. Week
   - Homogenous/inhomogenous points
   - Pinhole model
   - Camera metrix
   - Rigid transformation
   - Projection matrix
2. Week
   - Camera parameters
   - Lens distortion (radial distortion)
   - Homography estimation
3. Week
   - Epipolar lines
   - Fundamental matrix
   - Essential matrix
   - Triangulation
4. Week
   - Direct linear transform camera calibration
   - Zhang's checkerboard camera calibration
5. Week
   - Non-linear optimization
   - Levenberg-Marquardt algorithm (solving non-linear optimization problems)
   - Rotation parameterization of rotation for optimization
6. Week
   - Simple features
   - Image correspondence problem
   - Image features
   - Image filtering
   - Harris corner detection
   - Canny edge detection
7. Week
   - Hough transform
   - RANSAC algorithm
8. Week
   - BLOB detection using Difference-of-Gaussians (DoG)1
   - SIFT features
   - Feature matching
9. Week
   - Eight point algorithm for estimating fundamental matrix
   - Fundamental matrix with RANSAC
0. Week
   - RANSAC for fitting homographies
   - Stitching panoramas
1. Week
   - Essential matrix decomposition
   - Translation scale unceartanty
   - Perspective-$n$-Point problem
   - Visual odometry
2. Week
   - Guest lecture from Trackman
3. Week


# TODO
 - Fix extrinsics for week 4 Zhang


# Week 1
**Inhomogenous coordinates** "regular coordinates".  
**Homogeneous coordinates** concats a new constant dim 1 for ease of math.

When projecting from a higher dim space into a lower one, the higher dim inhom coord, can be treated as a lower dim homo point.

**Lines**
$$
0 = x + 2y - 3 = 
\left[
\begin{array}{c}
1 \\
2 \\
-3
\end{array}
\right] ^ T
\left[
\begin{array}{c}
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


# Week 2

**Skewed camera matrix**

$$
\mathbf{K} = \left[
\begin{array}{ccc}
f & \beta f & \delta_x \\
0 & \alpha f & \delta_y \\
0 & 0 & 1
\end{array}
\right]
$$

$\alpha$ ratio of focal lengths f_y / f_x. $\alpha \neq 1$ means non-square pixels.

$\beta$ skew factor with slope = beta / alpha. $\beta \neq 0$ means non-orthogonal axes, or non rectangular pixels.


**Radial distortion equations**

$$
\left[
\begin{array}{c}
x_d \\
y_d
\end{array}
\right]
 = dist \left(
\left[
\begin{array}{c}
x \\
y
\end{array}
\right]
 \right)
=
\left[
\begin{array}{c}
x \\
y
\end{array}
\right]
\cdot \left(1 + \Delta r(\sqrt{x^2 + y^2})\right)
$$
Where $\Delta r(r)$ is a polynomial with no $0^{th}$ and $1^{st}$-order term.

Regular projection
$$p = K[R|t]Q$$

Projection with distortion
$$p_d = K\Pi^{-1}(dist(\Pi([R|t]Q)))$$


**Homography** Maps one view of a plane to another.


# Week 3


**Epipolar lines**  
A point seen in one camera spans a line which can be seen in another camera.

A point seen from two different cameras speens two different lines. This two lines forms an **epipolar plane**.

The intersection of the epipolar plane and image plane is the **epipolar line**. Found using $l=Fp_1$.

**Epipole** is where all the epilolar lines from one camera intersect. It is the other cameracenter projected on the imageplane.

$q$ 2d point in pixels  
$p$ 3d point in camera frame of reference  
$$q = Kp$$
$$p = K^{-1}q$$


**Essential matrix**
Where $p_1$ is a point on the camera plane.

$$E=[t]_\times R$$

Where the normal to the plane is $n=Ep1$

$$p_2^TEp_1=0$$

Has 5 degrees of freedom

**Fundamental matrix**  
$$F=K_2^{-T}EK_1^{-1}$$
$$q_2^TFq_1=0$$

Has 7 degrees of freedom.


**Triangulation**
Using SVD minimizes a linear problem. Error is not optimal, as it weights error by the scale (the point measured from a camera far away has a largers impact on the error).


# Week 5

**Non linear optimization**  
$$e(x) = ||g(x) - y||_2^2 = f(x)^Tf(x)$$

**Euler angle**  
Rotate the object for each direction serially. Risks gimbal lock, meaning two parameters will rotate the same direction in certain situations (singularitys).

Works well if initial guess is good -> parameters will stay close to 0.

When using finite differentation, parameterize rotation of 0 as $2\pi$


**Axis angel**  
Rotation around ortogonal unit axis.

Convert to rotation matrix with `cv2.Rodrigues`. Is singular at **0**.


**Quanternions**  
Basically a complex number in 4D instead of 2D. Subject to $||q||_2=1$, has to be ajusted for when optimizing.


 
<!-- 
$$
\left[
\begin{array}{c}
 \\

\end{array}
\right]
$$
-->