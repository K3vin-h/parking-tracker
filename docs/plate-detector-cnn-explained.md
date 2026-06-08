# Plate Detector CNN Explained From Zero

This document explains how the project's plate detector works for someone who
has no background in machine learning, neural networks, or computer vision.

The goal is to make this file useful as a learning guide, not just as a short
technical reference.

Source file:

```text
apps/cv/models/plate_detector.py
```

Training script:

```text
apps/cv/training/train_detector.py
```

Tests:

```text
apps/cv/tests/test_plate_detector.py
```

## What Problem Are We Solving?

The parking tracker needs to read license plates from uploaded images.

That sounds like one task, but the project splits it into two smaller tasks:

```text
Task 1: Find where the plate is in the image.
Task 2: Read the letters and numbers on that plate.
```

`PlateDetectorCNN` does Task 1.

It looks at a full parking-lot image and predicts a rectangle around the
license plate.

It does not read the plate text.

The detector answers this question:

```text
Where is the license plate?
```

The recognizer answers this later question:

```text
What does the license plate say?
```

Splitting the work makes the full problem easier. Reading plate text from a
full image is hard because the image contains cars, pavement, shadows, lights,
and background objects. If the detector first crops the image down to just the
plate area, the recognizer has a much cleaner job.

## The Full Pipeline

The detector sits in the middle of the computer vision pipeline.

```text
uploaded image
-> validate image
-> resize image for detector
-> normalize pixels
-> convert image to tensor
-> PlateDetectorCNN predicts box
-> crop the plate region
-> PlateRecognizerCRNN reads text
-> parking app opens or closes a session
```

In plain English:

1. The user uploads an image.
2. The app checks that the image is safe and supported.
3. The image is resized to the detector's expected format.
4. The image is converted into numbers.
5. The detector predicts where the plate is.
6. The app crops that area from the image.
7. The recognizer reads the cropped plate.

This file focuses only on step 5.

## What Is Machine Learning?

Normal programming usually means writing exact rules.

For example:

```text
If the user clicks this button, save the form.
If the password is wrong, show an error.
If the session is active, calculate the price.
```

Those rules are clear because a developer can describe exactly what should
happen.

Finding a license plate in an image is harder to write as exact rules.

You could try rules like:

```text
Find a white rectangle.
Find dark letters inside it.
Find the rectangle near the back of a car.
```

But those rules break quickly.

License plates can be:

- bright or dark
- clean or dirty
- close or far away
- tilted
- partly shadowed
- on different colored cars
- photographed in different lighting

Machine learning solves this differently.

Instead of writing every visual rule by hand, we show the model many examples:

```text
image -> correct plate location
image -> correct plate location
image -> correct plate location
```

The model learns patterns from those examples.

After training, it can make a prediction for a new image it has never seen
before.

## What Is a Neural Network?

A neural network is a large math function with many adjustable numbers.

Those adjustable numbers are called parameters or weights.

At the beginning, the weights are mostly random. The network does not know how
to find plates yet.

Training slowly changes the weights so the network's predictions become closer
to the correct answers.

You can think of the model like this:

```text
input image numbers
-> many adjustable calculations
-> predicted plate box
```

The model is not memorizing one single rule. It is learning many small visual
patterns that work together.

## What Is Computer Vision?

Computer vision means using software to understand images.

Computers do not see an image the way humans do.

A computer sees an image as a grid of numbers.

For a color image, every pixel usually has three values:

```text
red, green, blue
```

For example, one pixel might be:

```text
[120, 180, 240]
```

That means:

- red value is 120
- green value is 180
- blue value is 240

Each value is usually between `0` and `255`.

The model does not receive "a car" or "a license plate" directly. It receives
millions of pixel numbers and must learn which number patterns usually mean
"license plate is here."

## What Is a Tensor?

In PyTorch, images are stored as tensors.

A tensor is just a container for numbers with a shape.

For this detector, the input shape is:

```text
(B, 3, H, W)
```

That looks scary at first, but each part has a simple meaning.

| Part | Meaning |
|------|---------|
| `B` | Batch size, or how many images are processed at once |
| `3` | Color channels: red, green, blue |
| `H` | Image height in pixels |
| `W` | Image width in pixels |

The usual detector input is:

```text
(B, 3, 480, 640)
```

That means:

```text
B images
3 color channels
480 pixels tall
640 pixels wide
```

If `B = 4`, the model is processing 4 images at once:

```text
(4, 3, 480, 640)
```

## Why Normalize Pixels?

Raw image pixels usually range from `0` to `255`.

The detector expects pixel values from `0` to `1`.

So this:

```text
0 to 255
```

becomes this:

```text
0.0 to 1.0
```

Example:

```text
255 becomes 1.0
128 becomes about 0.502
0 becomes 0.0
```

Neural networks usually train more reliably when the numbers are smaller and
more consistent. Normalizing pixels gives the model a stable input range.

## What Does the Detector Output?

The detector outputs four numbers:

```text
[cx, cy, w, h]
```

These four numbers describe a rectangle around the license plate.

| Value | Meaning |
|-------|---------|
| `cx` | Center x position of the plate |
| `cy` | Center y position of the plate |
| `w` | Width of the plate |
| `h` | Height of the plate |

These values are normalized between `0` and `1`.

That means the numbers are percentages of the full image size, not raw pixel
coordinates.

Example:

```text
[0.50, 0.60, 0.22, 0.08]
```

This means:

- `cx = 0.50`: the plate center is halfway across the image
- `cy = 0.60`: the plate center is 60% down from the top
- `w = 0.22`: the plate is 22% of the image width
- `h = 0.08`: the plate is 8% of the image height

This format is often called YOLO-style bounding-box format.

Important: this project uses the YOLO-style box format, but it does not use the
YOLO model architecture. The model is a custom CNN that predicts one plate box.

## Why Use Normalized Box Values?

Imagine one image is `640 x 480`, and another image is `1280 x 960`.

If the model predicted raw pixel values, the same plate position would need
different numbers for different image sizes.

Normalized values avoid that.

Halfway across the image is always:

```text
0.50
```

Whether the image is 640 pixels wide or 1280 pixels wide, `0.50` means the
center of the image.

This makes training and inference simpler.

## What Is a CNN?

CNN stands for Convolutional Neural Network.

A CNN is a type of neural network that is especially good at images.

Images have spatial structure:

- pixels next to each other matter
- edges are local patterns
- shapes are made of nearby edges
- objects are made of nearby shapes

A CNN is designed to learn from this structure.

Instead of looking at every pixel as a totally separate number, a CNN scans
small areas of the image and learns visual patterns.

For the plate detector, the CNN learns patterns like:

- horizontal edges
- vertical edges
- corners
- rectangular borders
- bright plate areas
- dark text-like marks
- plate-shaped regions on cars

## The Detector Architecture In One Picture

Here is the model from beginning to end:

```text
Input image tensor
(B, 3, H, W)

-> CNN Block 1
   Conv2d(3 -> 32)
   BatchNorm2d
   ReLU
   MaxPool2d

-> CNN Block 2
   Conv2d(32 -> 64)
   BatchNorm2d
   ReLU
   MaxPool2d

-> CNN Block 3
   Conv2d(64 -> 128)
   BatchNorm2d
   ReLU
   MaxPool2d

-> AdaptiveAvgPool2d(4 x 4)

-> Flatten

-> Linear(2048 -> 256)
-> ReLU
-> Dropout(0.3)

-> Linear(256 -> 4)
-> Sigmoid

-> [cx, cy, w, h]
```

The model has three main parts:

1. CNN backbone: learns image features
2. adaptive pooling: makes the feature size fixed
3. regression head: turns features into box coordinates

## What Is a Feature?

A feature is a useful pattern the model has learned.

Humans might describe image features with words:

```text
edge
corner
rectangle
plate border
dark text
```

The model stores features as numbers.

Early layers learn simple features. Later layers combine simple features into
more meaningful features.

For example:

```text
edges -> corners -> rectangles -> plate-like region
```

This is why CNNs use multiple layers. Each layer builds on the layer before it.

## CNN Block Pattern

Each detector block uses the same pattern:

```text
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
```

This pattern is common in CNNs because each piece has a clear role.

## Conv2d Explained

`Conv2d` is the main image-pattern-learning layer.

It uses small filters, also called kernels.

In this project, each convolution uses a `3 x 3` kernel.

That means the filter looks at a tiny 3-pixel-by-3-pixel area at a time.

```text
pixel pixel pixel
pixel pixel pixel
pixel pixel pixel
```

The filter slides across the image and produces a new grid of numbers.

At first, these filters are random. During training, they change until they
become useful.

One filter might become good at finding vertical edges. Another might become
good at finding bright-to-dark transitions. Another might respond to corners.

The model does not receive those meanings from us. It discovers useful filters
by trying to reduce prediction error during training.

## What Does `3 -> 32` Mean?

The first convolution is:

```text
Conv2d(3 -> 32)
```

This means:

```text
input channels: 3
output channels: 32
```

The input has 3 channels because it is an RGB image:

```text
red, green, blue
```

The output has 32 channels because the layer learns 32 different visual filters.

You can think of it like this:

```text
Original RGB image
-> 32 different pattern maps
```

Those pattern maps are not normal images anymore. They are learned feature maps.

## What Is Padding?

The convolutions use:

```text
padding=1
```

A `3 x 3` filter needs neighboring pixels. At the edge of an image, there are
not enough neighboring pixels unless we add padding.

Padding adds a border around the image so the convolution can process edge
pixels cleanly.

With `kernel_size=3` and `padding=1`, the convolution keeps the height and
width the same before pooling.

That makes the shape changes easier to reason about.

## Why `bias=False` In Conv Layers?

The model uses:

```text
bias=False
```

in its convolution layers.

That is because every convolution is immediately followed by `BatchNorm2d`.

Batch normalization already has learnable values that can shift the output up
or down. A separate convolution bias would be mostly redundant.

So the model skips the convolution bias to avoid unnecessary parameters.

## BatchNorm2d Explained

`BatchNorm2d` means batch normalization for 2D image feature maps.

During training, numbers inside a neural network can become unstable. One layer
might output values that are much larger or much smaller than the next layer
expects.

Batch normalization helps keep those values in a more stable range.

This usually helps training because:

- the model learns faster
- the model is less sensitive to random starting weights
- training is less chaotic
- it adds a small regularizing effect

For this project, that regularizing effect is useful because detector training
uses synthetic data. Synthetic data is helpful, but it can be less visually
varied than real-world images.

## ReLU Explained

`ReLU` stands for Rectified Linear Unit.

It is a simple function:

```text
if the number is negative, turn it into 0
if the number is positive, keep it
```

Examples:

```text
-3 becomes 0
-0.5 becomes 0
0 becomes 0
2 becomes 2
8 becomes 8
```

Why do this?

Because neural networks need non-linearity.

Without non-linear functions like ReLU, stacking many layers would not give the
model enough power to learn complex image patterns.

ReLU is popular because it is simple, fast, and works well in many CNNs.

The code uses:

```text
inplace=True
```

That means PyTorch can reuse memory instead of creating a brand-new tensor for
the ReLU result. This saves memory.

## MaxPool2d Explained

`MaxPool2d` shrinks feature maps.

The detector uses:

```text
MaxPool2d(kernel_size=2, stride=2)
```

That means it looks at each `2 x 2` area and keeps only the largest value.

Example:

```text
1  3
2  7
```

The max value is:

```text
7
```

So that `2 x 2` area becomes one number.

This cuts the height and width in half.

For the normal input size:

```text
480 x 640
-> after block 1: 240 x 320
-> after block 2: 120 x 160
-> after block 3: 60 x 80
```

Pooling helps because:

- it makes the model cheaper to run
- later layers can see larger parts of the original image
- small pixel-level changes matter less
- the model keeps the strongest feature signals

## Block 1: Low-Level Features

Block 1 is:

```text
Conv2d(3 -> 32)
BatchNorm2d(32)
ReLU
MaxPool2d(2 x 2)
```

Input:

```text
(B, 3, H, W)
```

Output:

```text
(B, 32, H/2, W/2)
```

For a `480 x 640` image:

```text
(B, 3, 480, 640)
-> (B, 32, 240, 320)
```

Block 1 learns low-level visual patterns:

- edges
- corners
- color changes
- bright and dark transitions

This is the foundation. Later blocks build more meaningful features from these
simple patterns.

## Block 2: Mid-Level Features

Block 2 is:

```text
Conv2d(32 -> 64)
BatchNorm2d(64)
ReLU
MaxPool2d(2 x 2)
```

Input:

```text
(B, 32, H/2, W/2)
```

Output:

```text
(B, 64, H/4, W/4)
```

For a `480 x 640` image:

```text
(B, 32, 240, 320)
-> (B, 64, 120, 160)
```

Block 2 combines simple patterns into larger shapes:

- lines
- borders
- rectangular outlines
- possible plate shapes

## Block 3: Higher-Level Features

Block 3 is:

```text
Conv2d(64 -> 128)
BatchNorm2d(128)
ReLU
MaxPool2d(2 x 2)
```

Input:

```text
(B, 64, H/4, W/4)
```

Output:

```text
(B, 128, H/8, W/8)
```

For a `480 x 640` image:

```text
(B, 64, 120, 160)
-> (B, 128, 60, 80)
```

Block 3 learns higher-level patterns:

- plate-like regions
- rectangles with inner text-like texture
- contrast between plate and car body
- plate-sized objects in plausible image areas

At this point, the model is no longer working with normal RGB pixels. It is
working with 128 learned feature maps.

## Why The Image Gets Smaller

The detector starts with a large image.

```text
480 x 640 = 307,200 pixel positions
```

Processing that at full size through every layer would be expensive.

Pooling shrinks the feature maps:

```text
480 x 640
-> 240 x 320
-> 120 x 160
-> 60 x 80
```

The model keeps useful information while reducing computation.

This is similar to how a person might first notice broad shapes in an image
instead of inspecting every pixel equally.

## AdaptiveAvgPool2d Explained

After the CNN blocks, the feature map size depends on the input image size.

For the standard input, the shape is:

```text
(B, 128, 60, 80)
```

But the fully connected layer later expects a fixed number of inputs.

That is where adaptive pooling helps.

The model uses:

```text
AdaptiveAvgPool2d((4, 4))
```

This always produces:

```text
(B, 128, 4, 4)
```

No matter whether the input image was:

```text
480 x 640
224 x 224
320 x 480
512 x 512
```

the adaptive pooling layer still outputs a `4 x 4` feature grid.

## Why Not Pool Down To 1 x 1?

The model could use global average pooling to make the feature map:

```text
(B, 128, 1, 1)
```

But that would remove too much location information.

The detector needs to know where the plate is, not just whether plate-like
features exist somewhere.

A `4 x 4` grid keeps rough spatial information:

```text
top-left     top      top-right
middle-left  middle   middle-right
bottom-left  bottom   bottom-right
```

It is not pixel-perfect location information, but it gives the regression head
enough spatial structure to predict a box.

## Flatten Explained

After adaptive pooling, the tensor shape is:

```text
(B, 128, 4, 4)
```

The model then flattens it.

Flattening means turning many dimensions into one long list of numbers.

The math is:

```text
128 * 4 * 4 = 2048
```

So the shape becomes:

```text
(B, 2048)
```

Each image is now represented by 2048 learned feature values.

Those 2048 values summarize what the CNN found and roughly where it found it.

## Fully Connected Layers Explained

After flattening, the model uses fully connected layers.

In PyTorch these are called `Linear` layers.

The first one is:

```text
Linear(2048 -> 256)
```

This compresses 2048 feature values into 256 values.

The second one is:

```text
Linear(256 -> 4)
```

This turns the 256 values into the final four box numbers:

```text
[cx, cy, w, h]
```

This part is called the regression head.

## What Is Regression?

Regression means predicting continuous numbers.

The detector predicts coordinates, so this is regression.

Examples of regression:

```text
predict a price
predict a temperature
predict a distance
predict box coordinates
```

This is different from classification.

Classification means choosing a category.

Examples of classification:

```text
cat or dog
spam or not spam
red, blue, or green
entry or exit
```

The detector is not choosing a category. It is predicting exact numeric box
values.

## Dropout Explained

The model uses:

```text
Dropout(0.3)
```

Dropout randomly turns off some feature values during training.

With `0.3`, about 30% of the values are temporarily set to zero during each
training forward pass.

Why would we intentionally remove information?

Because it prevents the model from relying too heavily on one exact feature.

Without dropout, the model might memorize synthetic training details instead
of learning general plate patterns.

Dropout forces the model to spread knowledge across many features.

During evaluation and inference, dropout is disabled. That way the same input
produces the same output every time.

## Sigmoid Explained

The final layer produces four raw numbers.

Raw neural network outputs can be any value:

```text
-4.2
0.7
3.1
12.9
```

But bounding-box values should be between `0` and `1`.

The model applies sigmoid:

```text
torch.sigmoid(x)
```

Sigmoid squeezes any number into the range:

```text
0 to 1
```

Examples:

```text
large negative number -> close to 0
0 -> 0.5
large positive number -> close to 1
```

This is why the model's output can safely represent normalized coordinates.

Important: sigmoid is already applied inside `forward()`. Do not apply sigmoid
again outside the model.

Applying it twice would distort the prediction.

## `forward()` Explained

In PyTorch, `forward()` defines what happens when data passes through the
model.

The detector's `forward()` does this:

```text
input image
-> block1
-> block2
-> block3
-> adaptive pool
-> flatten
-> fc1
-> ReLU
-> dropout
-> fc2
-> sigmoid
-> box prediction
```

In code, calling the model automatically calls `forward()`:

```python
preds = model(images)
```

You usually do not call `model.forward(images)` directly.

## `predict()` Explained

The model also has a `predict()` method.

```python
preds = model.predict(images)
```

`predict()` is for inference, which means using the trained model to make
predictions.

It does three important things:

1. disables gradient tracking
2. switches the model into eval mode
3. restores the previous mode afterward

## What Are Gradients?

Gradients are numbers used during training to figure out how the model's
weights should change.

During inference, we are not training. We only want predictions.

So `predict()` uses:

```python
@torch.no_grad()
```

This tells PyTorch:

```text
Do not track training information for this prediction.
```

That makes inference faster and uses less memory.

## What Is Eval Mode?

PyTorch models can be in train mode or eval mode.

Train mode is for learning.

Eval mode is for stable prediction.

This matters because some layers behave differently in train mode and eval
mode.

For this detector, the important example is dropout:

```text
train mode: dropout randomly turns off features
eval mode: dropout is disabled
```

During prediction, we want stable results. So `predict()` temporarily switches
the model to eval mode.

## How Training Works

At the start, the model is bad at finding plates.

Training improves it through repeated practice.

One training step looks like this:

```text
1. Give the model a training image.
2. Model predicts [cx, cy, w, h].
3. Compare prediction to the correct box.
4. Calculate how wrong the model was.
5. Use that error to adjust the weights.
6. Repeat many times.
```

The "how wrong" number is called loss.

The goal of training is to reduce loss.

## Where The Training Data Comes From

The detector is trained with synthetic data.

The project generates images by placing fake rendered license plates onto
parking-lot background photos.

Because the generator placed the plate, it knows the exact correct box.

So each training example has:

```text
image
correct bounding box
```

This is useful because hand-labeling thousands of real license plate images
would take a long time.

Synthetic data lets the project create many labeled examples automatically.

## What Is a Label?

A label is the correct answer for a training example.

For this detector, the label is the correct plate box:

```text
[cx, cy, w, h]
```

The model makes a prediction, then training compares the prediction to this
label.

## What Is a Batch?

Instead of training on one image at a time, the model usually trains on a batch
of images.

If the batch size is 32, one step uses:

```text
32 images
32 correct boxes
32 predicted boxes
```

Batches make training more efficient and give the optimizer a more stable view
of the data.

## Loss Function Explained

The detector uses:

```text
SmoothL1Loss
```

A loss function measures how wrong the prediction is.

For example:

```text
prediction: [0.50, 0.60, 0.22, 0.08]
label:      [0.52, 0.58, 0.20, 0.09]
```

The loss function compares those numbers and returns one error value.

Lower loss means the prediction is closer to the label.

## Why Smooth L1 Loss?

Smooth L1 loss is also called Huber loss.

It is useful for bounding boxes because it handles both small and large
mistakes well.

Mean squared error can punish large mistakes too strongly, especially early in
training when the model is still bad.

Plain absolute error is more stable for large mistakes, but it can be less
smooth near the correct answer.

Smooth L1 is a compromise:

- for small errors, it behaves smoothly
- for large errors, it does not explode

That makes training more stable.

## Optimizer Explained

The training script uses:

```text
Adam
```

The optimizer is the part that updates the model's weights.

The loss function says:

```text
The model was this wrong.
```

The optimizer says:

```text
Change the weights this way to become less wrong next time.
```

Adam is popular because it adapts update sizes for different weights. This
often works well for neural networks without needing as much manual tuning.

## Learning Rate Explained

The learning rate controls how big each training update is.

If the learning rate is too high:

```text
the model may jump around and fail to settle
```

If the learning rate is too low:

```text
the model may learn very slowly
```

The detector training script starts with a learning rate and then adjusts it
during training.

## Scheduler Explained

The detector training script uses:

```text
ReduceLROnPlateau
```

This scheduler watches validation loss.

If validation loss stops improving, it lowers the learning rate.

In plain English:

```text
If progress stalls, take smaller steps.
```

This helps the model improve carefully after the easy early progress is done.

## Gradient Clipping Explained

The training script uses gradient clipping:

```text
clip_grad_norm_(..., max_norm=1.0)
```

Sometimes gradients can become too large, especially early in training.

Large gradients can cause overly large weight updates.

Gradient clipping limits how large the update signal can become.

This helps keep training stable.

## Validation Explained

Training data teaches the model.

Validation data checks whether the model is learning in a way that works on
examples it is not directly training on.

The training script splits the dataset into:

```text
training split
validation split
```

The model updates its weights using the training split.

The model is evaluated on the validation split.

This helps detect overfitting.

## Overfitting Explained

Overfitting means the model memorizes training examples instead of learning a
general skill.

An overfit detector might do well on generated training images but poorly on
new images.

The project reduces overfitting with:

- synthetic variation
- image augmentation
- dropout
- validation checks
- saving the best validation model

## IoU Explained

The training script reports IoU.

IoU means Intersection over Union.

It measures how much the predicted box overlaps the correct box.

Imagine two rectangles:

```text
predicted box
correct box
```

The overlap is the area where both boxes cover the same pixels.

The union is the total area covered by either box.

The formula is:

```text
IoU = overlap area / union area
```

The score is between `0` and `1`.

| IoU | Meaning |
|-----|---------|
| `0.0` | The boxes do not overlap |
| `0.5` | The boxes partially overlap |
| `1.0` | The boxes match perfectly |

The project target is:

```text
validation IoU > 0.7
```

That means the predicted plate boxes should overlap the correct boxes well
enough to give the recognizer a useful crop.

## Why Loss And IoU Are Both Used

Loss and IoU tell different stories.

Loss measures coordinate error.

IoU measures box overlap.

A low loss usually helps, but IoU is easier to understand visually because it
answers:

```text
How much does the predicted rectangle overlap the true rectangle?
```

For a detector, overlap is what matters for cropping.

## Saving Weights Explained

When training finishes, the model saves learned weights to a file.

The planned detector weight path is:

```text
apps/cv/weights/detector.pth
```

Weights are the learned numbers inside the model.

The architecture is the code structure.

The weights are what the model learned during training.

You need both:

```text
architecture + trained weights = useful trained model
```

## Why This Model Is Small

This detector is intentionally small.

It does not need to find many object categories.

It only needs to find one kind of object:

```text
license plate
```

A large general-purpose detector would be more complex than needed for this
project stage.

This custom CNN is a good fit because:

- it predicts one box
- it is easier to understand
- it is fast to train compared with larger detectors
- synthetic data provides exact box labels
- the recognizer handles text reading separately

## What The Tests Protect

The detector tests make sure the model keeps the behavior the rest of the app
depends on.

The tests check that:

- a normal `480 x 640` input returns shape `(B, 4)`
- a batch size of `1` works
- different image sizes still work
- output values stay inside `[0, 1]`
- outputs are `float32`
- eval mode gives stable predictions
- train mode uses dropout
- `predict()` does not track gradients
- `predict()` does not apply sigmoid twice
- IoU works for perfect overlap
- IoU works for partial overlap
- IoU works for no overlap

These tests matter because a bad detector output can break the next pipeline
step. If the box is invalid, the crop can be wrong. If the crop is wrong, the
recognizer may read the wrong text or fail completely.

## A Simple Walkthrough With One Image

Imagine the app receives one image:

```text
one parking-lot photo
```

The image is resized and converted into a tensor:

```text
(1, 3, 480, 640)
```

The `1` means one image in the batch.

Block 1 scans for simple patterns:

```text
edges, corners, color changes
```

Block 2 combines them:

```text
lines, borders, rectangles
```

Block 3 combines them further:

```text
plate-like areas
```

Adaptive pooling compresses the result:

```text
(1, 128, 60, 80)
-> (1, 128, 4, 4)
```

Flatten turns it into:

```text
(1, 2048)
```

The fully connected layers turn those features into:

```text
(1, 4)
```

Example output:

```text
[[0.50, 0.60, 0.22, 0.08]]
```

The app converts that normalized box into a crop area and cuts the plate out of
the image.

Then the cropped plate goes to the recognizer.

## Common Confusions

### Does The Detector Read Letters?

No.

It only finds the plate location.

The recognizer reads the letters later.

### Is This YOLO?

No.

The output box format is YOLO-style:

```text
[cx, cy, w, h]
```

But the architecture is a custom CNN, not the YOLO object detection model.

### Why Only One Box?

The project currently expects one main plate in the uploaded image.

So the detector predicts one box.

If the app later needs to detect multiple plates in one image, the architecture
would need to change.

### Why Not Use An External API?

The project requirement is to build the CV pipeline from scratch with custom
models.

That is why the detector is implemented in PyTorch instead of calling an
external computer vision API.

### Why Does The Model Need Training?

The architecture defines the shape of the model.

Training teaches the model what values its weights should have.

An untrained detector has the right structure but does not know how to find
plates yet.

## Mental Model To Remember

The detector is like a student learning to point at a license plate.

At first, it guesses badly.

During training, it sees many images where the correct plate box is known.

Every time it guesses, the training loop measures the mistake and adjusts the
model.

Over many examples, the model learns visual patterns that help it predict:

```text
the plate is centered here
the plate is this wide
the plate is this tall
```

In code, that final answer is:

```text
[cx, cy, w, h]
```

## Short Summary

`PlateDetectorCNN` is a custom convolutional neural network that finds the
license plate in a full image.

It works by:

1. converting the image into numbers
2. using CNN blocks to learn visual patterns
3. shrinking the image features while keeping useful information
4. pooling features into a fixed `4 x 4` grid
5. flattening those features into 2048 numbers
6. using fully connected layers to predict four box values
7. applying sigmoid so the box values stay between `0` and `1`

The detector's output is not the plate text.

The detector's output is the plate location:

```text
[cx, cy, w, h]
```

That location is used to crop the plate before the recognizer reads it.
