# PR #7 Explained: PlateRecognizerCRNN

Pull request: https://github.com/K3vin-h/parking-tracker/pull/7

PR title: `feat: add PlateRecognizerCRNN model and recognizer training script`

## The Short Version

This PR adds the part of the parking tracker that tries to **read the letters and numbers from a cropped license plate image**.

Before this PR, the project had the detector:

```text
Full car / parking image -> find where the license plate is
```

This PR adds the recognizer:

```text
Cropped license plate image -> read the plate text
```

Together, the future computer vision flow becomes:

```text
1. User uploads a parking-lot image.
2. PlateDetectorCNN finds the license plate area.
3. The app crops that plate area out of the big image.
4. PlateRecognizerCRNN reads the cropped plate.
5. The app gets text like "ABC123".
6. The parking system opens or closes a parking session.
```

## What Is Machine Learning?

Machine learning, or ML, is a way to make a computer learn patterns from examples instead of writing every rule by hand.

Normal programming looks like this:

```text
Human writes exact rules -> computer follows those rules
```

Machine learning looks like this:

```text
Human gives many examples -> computer adjusts itself -> computer learns a pattern
```

For example, if you wanted to recognize the number `8`, it would be very hard to write rules like:

```text
If there are two loops and they are stacked, maybe it is an 8.
If the top loop is smaller, still maybe it is an 8.
If the image is blurry, maybe still an 8.
If the font is different, maybe still an 8.
```

That gets messy fast.

Instead, ML gives the computer thousands of examples:

```text
image of 8 -> correct answer is "8"
image of B -> correct answer is "B"
image of ABC123 -> correct answer is "ABC123"
```

The model makes guesses. When it guesses wrong, training changes the model slightly so it gets better next time.

## What Is A Model?

In this project, a **model** is a Python class built with PyTorch. It is a math machine with many adjustable numbers inside it.

Those adjustable numbers are called **parameters** or **weights**.

At the start, the weights are mostly random. The model is bad at reading plates.

During training:

```text
model guesses -> compare guess to correct answer -> calculate mistake -> adjust weights
```

After many examples, the weights should become useful.

The trained weights are saved into a file like:

```text
apps/cv/weights/recognizer.pth
```

That `.pth` file is the learned knowledge. The code defines the model shape, but the `.pth` file contains what the model learned.

## What Problem Does PR #7 Solve?

The detector model answers this question:

```text
Where is the license plate in this image?
```

The recognizer model answers this question:

```text
What letters and numbers are on this plate?
```

Example:

```text
Input to detector:
    full image of a car

Detector output:
    box around license plate

Input to recognizer:
    cropped image of just the plate

Recognizer output:
    "ABC123"
```

PR #7 adds the recognizer part.

## Files Changed In PR #7

PR #7 changes four files:

| File | What it does |
| --- | --- |
| `apps/cv/models/__init__.py` | Makes the new recognizer model importable |
| `apps/cv/models/recognizer.py` | Adds the actual plate-reading neural network |
| `apps/cv/training/train_recognizer.py` | Adds the script used to train the recognizer |
| `apps/cv/tests/test_plate_recognizer.py` | Adds tests that check the recognizer behaves correctly |

## Important Vocabulary

### Image Tensor

A tensor is just a grid of numbers.

Images are stored as numbers because computers do not understand pictures the way humans do.

A grayscale image has one number per pixel:

```text
0.0 = black
1.0 = white
values between 0 and 1 = shades of gray
```

The recognizer expects input shaped like:

```text
(B, 1, 32, 128)
```

That means:

| Part | Meaning |
| --- | --- |
| `B` | Batch size, how many plate images are processed at once |
| `1` | One color channel, because the image is grayscale |
| `32` | Image height, 32 pixels |
| `128` | Image width, 128 pixels |

So one plate image looks like:

```text
(1, 1, 32, 128)
```

A batch of 8 plate images looks like:

```text
(8, 1, 32, 128)
```

### Batch

A batch is a group of examples processed at the same time.

Instead of training on one image at a time:

```text
image 1
image 2
image 3
```

The model trains on a batch:

```text
images 1 through 32 at once
```

This is faster and makes training smoother.

### CNN

CNN means **Convolutional Neural Network**.

A CNN is good at looking at images.

It learns visual patterns in layers:

```text
early layers: edges, corners, lines
middle layers: curves, strokes, shapes
later layers: character-like patterns
```

For license plates, a CNN helps recognize visual pieces like:

```text
vertical line in "1"
round shape in "0"
diagonal lines in "A"
curves in "B"
```

### LSTM

LSTM means **Long Short-Term Memory**.

An LSTM is good at reading sequences.

A license plate is a sequence:

```text
A -> B -> C -> 1 -> 2 -> 3
```

The recognizer first uses a CNN to understand the image, then uses an LSTM to read the plate from left to right.

### Bidirectional LSTM

Bidirectional means it reads both directions:

```text
left to right
right to left
```

This helps because surrounding characters can give context.

Example:

```text
O and 0 can look similar.
I and 1 can look similar.
B and 8 can look similar.
```

Seeing nearby characters can help the model make a better guess.

### CRNN

CRNN means:

```text
Convolutional Recurrent Neural Network
```

In plain English:

```text
CNN reads the image.
LSTM reads the image features as a sequence.
```

That is why this model is called:

```python
PlateRecognizerCRNN
```

### CTC

CTC means **Connectionist Temporal Classification**.

The name is not important. The idea is important.

CTC lets the model read text from an image without needing to know exactly where each letter starts and ends.

That matters because a plate image does not come with perfect character boxes like:

```text
[A] [B] [C] [1] [2] [3]
```

The training data knows the final text:

```text
"ABC123"
```

But it usually does not label the exact pixel range of each character.

CTC handles that problem.

## The Recognizer Model Shape

The recognizer takes this:

```text
cropped grayscale plate image
shape: (B, 1, 32, 128)
```

It outputs this:

```text
shape: (T=16, N, C=37)
```

Where:

| Symbol | Meaning |
| --- | --- |
| `T=16` | 16 time steps, like 16 left-to-right positions across the plate |
| `N` | Batch size |
| `C=37` | 37 possible classes |

The 37 classes are:

```text
blank token
A through Z
0 through 9
```

The blank token is used by CTC. It means "no character here."

## Why 16 Time Steps?

The plate image is 128 pixels wide.

The CNN shrinks the width as it processes the image:

```text
128 pixels wide -> 64 -> 32 -> 16
```

At the end, the model has 16 vertical slices of information.

Think of it like the model scans the plate in 16 chunks:

```text
| 1 | 2 | 3 | 4 | 5 | 6 | ... | 16 |
```

Each chunk gets a chance to predict a character or blank.

This works because most plates are around 6 to 8 characters, so 16 time steps gives the model enough room.

## File 1: `apps/cv/models/__init__.py`

This file is small but useful.

It now imports:

```python
from apps.cv.models.plate_detector import PlateDetectorCNN
from apps.cv.models.recognizer import PlateRecognizerCRNN
```

And exports:

```python
__all__ = ["PlateDetectorCNN", "PlateRecognizerCRNN"]
```

That means other files can import both models from one place:

```python
from apps.cv.models import PlateDetectorCNN, PlateRecognizerCRNN
```

Why this matters:

```text
Cleaner imports
Less repeated path typing
Makes the CV model package feel complete
```

## File 2: `apps/cv/models/recognizer.py`

This is the core model file.

It creates:

```python
class PlateRecognizerCRNN(nn.Module):
```

`nn.Module` is PyTorch's base class for neural networks.

### The Model Has Three Main Parts

```text
1. CNN backbone
2. Bidirectional LSTM
3. Output layer + CTC decoder helper
```

### Part 1: CNN Backbone

The CNN backbone has 3 blocks.

Each block follows this pattern:

```text
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
```

Here is what each piece means:

| Piece | Beginner explanation |
| --- | --- |
| `Conv2d` | Looks for visual patterns in the image |
| `BatchNorm2d` | Keeps numbers stable while training |
| `ReLU` | Keeps positive useful signals and removes negative ones |
| `MaxPool2d` | Shrinks the image while keeping strong features |

The image shape changes like this:

```text
Input:        (B, 1,   32, 128)
After block1: (B, 64,  16, 64)
After block2: (B, 128,  8, 32)
After block3: (B, 256,  8, 16)
```

Plain English:

```text
The image gets smaller in width and height.
The number of learned feature channels gets larger.
```

At the start, there is 1 grayscale channel.

At the end, there are 256 feature channels.

Those 256 channels are not colors. They are learned pattern detectors.

### Why Keep Height At 8 In The Last Block?

The third block uses:

```python
MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
```

That means:

```text
do not shrink height
shrink width
```

Why?

Letters need vertical detail.

For example:

```text
I vs 1
O vs 0
B vs 8
```

If the model shrinks the height too much, it may lose details that help distinguish similar characters.

### Part 2: Reshape Image Features Into A Sequence

After the CNN, the tensor shape is:

```text
(B, 256, 8, 16)
```

That means:

```text
B = batch size
256 = learned feature channels
8 = height
16 = width positions
```

The model reshapes it into:

```text
(16, B, 2048)
```

Why 2048?

```text
256 channels * 8 height rows = 2048 numbers per width position
```

So each of the 16 width positions becomes one time step:

```text
time step 1 has 2048 features
time step 2 has 2048 features
...
time step 16 has 2048 features
```

That turns the image problem into a sequence problem.

### Part 3: Bidirectional LSTM

The LSTM reads the 16 time steps.

The code uses:

```python
nn.LSTM(
    input_size=2048,
    hidden_size=256,
    num_layers=2,
    bidirectional=True,
)
```

Plain English:

| Setting | Meaning |
| --- | --- |
| `input_size=2048` | Each time step has 2048 numbers from the CNN |
| `hidden_size=256` | The LSTM stores 256 learned memory values per direction |
| `num_layers=2` | Two LSTM layers stacked together |
| `bidirectional=True` | Reads both left-to-right and right-to-left |

Because it is bidirectional, the output size becomes:

```text
256 forward + 256 backward = 512
```

So after the LSTM:

```text
(16, B, 512)
```

### Part 4: Final Output Layer

The final layer is:

```python
self.fc = nn.Linear(512, VOCAB_SIZE)
```

`VOCAB_SIZE` is 37.

So it turns each time step from 512 numbers into 37 scores:

```text
score for blank
score for A
score for B
...
score for Z
score for 0
...
score for 9
```

Then the model applies:

```python
F.log_softmax(x, dim=-1)
```

That turns raw scores into log probabilities.

## What Is Log Softmax?

This is a math detail, but here is the simple version.

The model starts by producing raw scores:

```text
A: 2.1
B: 0.3
C: -1.5
...
```

Those are not probabilities yet.

Softmax turns them into probabilities:

```text
A: 0.70
B: 0.20
C: 0.02
...
```

All probabilities add up to 1.

`log_softmax` stores those probabilities in log form because `CTCLoss` expects log probabilities.

Important rule:

```text
Do not apply log_softmax twice.
```

The model already does it inside `forward()`.

## `forward()` Method

The `forward()` method is what runs when you call:

```python
output = model(images)
```

It does this:

```text
1. pass image through CNN blocks
2. reshape CNN output into 16 time steps
3. pass sequence through BiLSTM
4. pass LSTM output through final linear layer
5. apply log_softmax
6. return shape (16, batch, 37)
```

## `predict()` Method

`predict()` is for inference, meaning using the model to make a prediction.

It does this:

```text
1. remembers whether the model was in training mode
2. switches to eval mode
3. runs forward()
4. restores the old mode afterward
```

Why this matters:

During training, some layers behave randomly on purpose. Dropout is one example.

During prediction, we want stable results.

So `predict()` temporarily switches to eval mode.

It also uses:

```python
@torch.no_grad()
```

That means PyTorch will not save information needed for training gradients.

This makes prediction:

```text
faster
less memory-heavy
safer for inference
```

## `decode_predictions()` Method

The model output is not directly text.

It is a big tensor of log probabilities.

`decode_predictions()` turns that tensor into strings.

It uses greedy CTC decoding.

### Step 1: Pick The Most Likely Class At Each Time Step

Example:

```text
time 1 -> blank
time 2 -> A
time 3 -> A
time 4 -> blank
time 5 -> B
time 6 -> C
time 7 -> blank
time 8 -> 1
time 9 -> 2
time 10 -> 3
```

### Step 2: Collapse Repeated Tokens

CTC may repeat the same character across multiple time steps.

Example:

```text
A, A, A
```

usually means:

```text
A
```

not:

```text
AAA
```

So repeated neighboring tokens are collapsed.

### Step 3: Remove Blanks

Blanks are not real characters.

So:

```text
blank, A, blank, B, C, blank, 1, 2, 3
```

becomes:

```text
ABC123
```

## Why Blanks Are Useful

Blanks let the model separate repeated real characters.

Example:

Suppose the real plate is:

```text
AA123
```

Without blanks, repeated `A` values might get collapsed into one `A`.

CTC can represent repeated letters like this:

```text
A, blank, A, 1, 2, 3
```

After decoding:

```text
AA123
```

So blanks are not noise. They are part of how CTC works.

## File 3: `apps/cv/training/train_recognizer.py`

This file trains the recognizer model.

Training means:

```text
show the model many plate images
let it guess the text
compare guess to correct text
adjust the model weights
repeat many times
```

## How To Run Training

First generate synthetic recognizer data:

```bash
python -c "
from apps.cv.training.synthetic_data import generate_recognizer_dataset
generate_recognizer_dataset(n=5000, output_dir='data/recognizer')
"
```

Then train:

```bash
python apps/cv/training/train_recognizer.py \
    --data-dir data/recognizer \
    --epochs 100 \
    --output apps/cv/weights/recognizer.pth
```

## What Is Synthetic Data?

Synthetic data means fake training examples made by code.

Instead of taking thousands of real license plate photos, the project can generate plate images.

A synthetic example might include:

```text
image: generated plate crop
label: "ABC123"
```

The benefit:

```text
cheap to make
can generate thousands
exact labels are known
good for learning the first version of the model
```

The downside:

```text
fake images may not perfectly match real camera photos
```

Eventually, real-world testing will still matter.

## Training Script Flow

The training script does this:

```text
1. Parse command-line arguments.
2. Pick device: MPS, CUDA, or CPU.
3. Load PlateRecognizerDataset.
4. Split data into training and validation sets.
5. Create DataLoaders.
6. Create PlateRecognizerCRNN.
7. Create CTCLoss.
8. Create Adam optimizer.
9. Train for many epochs.
10. Validate after each epoch.
11. Save best weights.
12. Save a training progress chart.
```

## Device Selection

The script calls:

```python
get_device()
```

That chooses the best available hardware:

```text
MPS on Apple Silicon Macs
CUDA on NVIDIA GPUs
CPU if no GPU backend is available
```

## Why CTCLoss Runs On CPU For MPS

Apple Silicon uses a PyTorch backend called MPS.

The model can run on MPS, but PyTorch does not support `CTCLoss` on MPS.

So the code does:

```python
loss = criterion(log_probs.cpu(), targets, input_lengths, target_lengths)
```

Plain English:

```text
Run the model on the fast device.
Move the output to CPU for the CTC loss calculation.
Still let PyTorch calculate gradients correctly.
```

This is a workaround so training does not crash on Mac.

## What Is Loss?

Loss is a number that says how wrong the model is.

High loss:

```text
model is very wrong
```

Low loss:

```text
model is closer to correct
```

Training tries to reduce loss over time.

## What Is An Epoch?

One epoch means the model has seen the whole training dataset once.

If you have 5,000 images:

```text
1 epoch = model trained across all 5,000 images once
100 epochs = model went through the dataset 100 times
```

## What Is An Optimizer?

The optimizer is the part that adjusts model weights after each mistake.

This script uses:

```python
torch.optim.Adam
```

Adam is a popular optimizer because it usually works well without needing too much manual tuning.

## What Is Learning Rate?

Learning rate controls how big each weight update is.

If learning rate is too high:

```text
the model may bounce around and fail to learn
```

If learning rate is too low:

```text
the model may learn too slowly
```

This script starts with:

```text
0.001
```

That is written as:

```text
1e-3
```

## What Is ReduceLROnPlateau?

The script uses:

```python
ReduceLROnPlateau
```

Plain English:

```text
If validation loss stops improving, reduce the learning rate.
```

This helps when training gets stuck.

## What Is Validation Data?

The script splits the dataset:

```text
80% training
20% validation
```

Training data is used to update the model.

Validation data is used to check whether the model is actually learning, not just memorizing.

If training loss improves but validation loss gets worse, that can mean the model is overfitting.

## What Is Overfitting?

Overfitting means the model memorizes the training examples instead of learning the general pattern.

Example:

```text
Student memorizes answers from one practice test.
Student fails when questions are slightly different.
```

For ML:

```text
model does well on training images
model does badly on new images
```

Validation data helps catch this.

## What The Training Script Measures

The script tracks:

```text
training loss
validation loss
character accuracy
plate accuracy
learning rate
```

### Character Accuracy

Character accuracy checks individual characters.

Example:

```text
Correct plate: ABC123
Predicted:     ABC128
```

The model got 5 out of 6 characters right:

```text
A correct
B correct
C correct
1 correct
2 correct
3 wrong, predicted 8
```

So character accuracy is high, even though the whole plate is not perfect.

### Plate Accuracy

Plate accuracy checks whether the entire plate is exactly correct.

Example:

```text
Correct plate: ABC123
Predicted:     ABC128
```

Plate accuracy for this example is wrong because one character is incorrect.

The PR goal says:

```text
>90% character accuracy
>80% full-plate exact match
```

## Best Weights Checkpointing

The script saves the model when validation loss improves.

That means it does not simply save the final epoch.

It saves the best version seen during training.

Example:

```text
epoch 1: validation loss 5.2
epoch 2: validation loss 4.1 -> save
epoch 3: validation loss 3.4 -> save
epoch 4: validation loss 3.8 -> do not save
```

The best saved model would be from epoch 3.

## Training Progress Chart

The script creates a chart with 4 panels:

```text
1. Loss
2. Character accuracy
3. Plate accuracy
4. Learning rate
```

This chart helps you see whether training is working.

Good signs:

```text
loss goes down
character accuracy goes up
plate accuracy goes up
learning rate drops when progress slows
```

Bad signs:

```text
loss stays flat
accuracy stays near zero
validation gets worse while training improves
```

## Output Path Safety

The training script checks that the output file stays inside the project.

This prevents unsafe paths like:

```text
../../../../somewhere/outside/project
```

That matters because training scripts write files. File-writing code should be careful about where it writes.

## File 4: `apps/cv/tests/test_plate_recognizer.py`

This file adds tests for the recognizer.

Tests are important because ML code can break in ways that still "run."

For example:

```text
wrong output shape
wrong softmax dimension
predict() accidentally leaves model in eval mode
decoder removes repeated letters incorrectly
```

Those bugs might not crash immediately, but they can ruin training.

## What The Tests Check

### Output Shape

The model must output:

```text
(16, batch_size, 37)
```

The tests check normal batches and batch size 1.

Batch size 1 matters because some code accidentally removes dimensions when there is only one item.

### Log Softmax Validity

The tests check that:

```text
all log probabilities are <= 0
exp(output) sums to 1 across the class dimension
```

This proves the model output is valid for `CTCLoss`.

### `predict()` Behavior

The tests check:

```text
predict() does not track gradients
predict() restores training mode if model was training
predict() keeps eval mode if model was already eval
```

This matters because `predict()` may be used during training for quick checks.

It should not accidentally change the training loop.

### Determinism In Eval Mode

Eval mode should be stable:

```text
same input -> same output
```

The tests check this.

### Dropout In Train Mode

Train mode uses dropout, so repeated runs can produce different outputs.

That is expected.

Dropout helps reduce overfitting.

### Parameter Count

The test checks the model has a reasonable number of trainable parameters:

```text
between 5 million and 20 million
```

Why?

If the count is way too low:

```text
maybe a layer is missing
```

If the count is way too high:

```text
maybe the architecture accidentally exploded in size
```

### Decoder Edge Cases

The tests check CTC decoding rules:

```text
blank tokens are removed
repeated tokens are collapsed
blank can separate repeated real characters
all-blank output becomes empty string
decoded result is always a string
```

These are important because the decoder turns math output into real plate text.

## Example: How Decoding Works

Imagine the model predicts this sequence:

```text
blank, A, A, blank, B, C, blank, 1, 2, 3
```

First, collapse repeated tokens:

```text
blank, A, blank, B, C, blank, 1, 2, 3
```

Then remove blanks:

```text
A, B, C, 1, 2, 3
```

Final result:

```text
ABC123
```

Now imagine the real text has repeated letters:

```text
AA123
```

The model can output:

```text
A, blank, A, 1, 2, 3
```

Collapse repeated tokens:

```text
A, blank, A, 1, 2, 3
```

Remove blanks:

```text
AA123
```

That is why blanks matter.

## How PR #7 Fits With The Previous PRs

Earlier work added:

```text
image preprocessing
synthetic data generation
dataset classes
plate detector model
detector training script
```

PR #7 adds:

```text
plate recognizer model
recognizer training script
recognizer tests
```

The project is building the CV pipeline in order:

```text
Step 1: prepare images
Step 2: generate training data
Step 3: train detector to find plates
Step 4: train recognizer to read plates
Step 5: connect detector + recognizer into app inference
Step 6: use result in parking session logic
```

PR #7 completes Step 4 at the code level.

## What This PR Does Not Do Yet

This PR does not fully connect recognition to the web app.

It does not yet:

```text
run recognition from the upload API
open or close parking sessions from recognized text
show recognition results in the dashboard
handle manual corrections in the UI
train real production weights
prove accuracy on real camera images
```

It adds the model, training script, and tests needed before those later steps.

## Mental Model To Remember

Think of the system like a student learning to read license plates.

At first, the student knows nothing.

Training gives the student many examples:

```text
picture -> correct answer
picture -> correct answer
picture -> correct answer
```

The student guesses, checks the answer, and adjusts.

The CNN part learns what letters and numbers look like.

The LSTM part learns how to read them in order from left to right.

The CTC part cleans up the messy sequence into final text.

So the recognizer is basically:

```text
look carefully at the image
scan across it
guess characters at each position
clean up the guesses into plate text
```

## Final Beginner Summary

PR #7 adds the license plate reader.

The model is called `PlateRecognizerCRNN`.

It takes a small grayscale image of a plate:

```text
32 pixels tall, 128 pixels wide
```

It turns that image into 16 left-to-right steps.

At each step, it guesses one of:

```text
blank, A-Z, 0-9
```

Then CTC decoding removes blanks and repeated predictions to produce final text like:

```text
ABC123
```

The PR also adds a training script so the model can learn from synthetic plate images, and it adds tests to make sure the model output, prediction method, and decoder all behave correctly.

---

# Deep Beginner Walkthrough: What Every Important PR #7 File Is Doing

This section is meant for a student reading machine learning code for the first
time. It explains the code slowly, in plain language, and connects each file to
the bigger idea.

The most important thing to remember:

```text
The model does not "know" letters by magic.
It receives an image as numbers.
It pushes those numbers through layers.
Each layer changes the numbers a little.
At the end, the model produces guesses.
Training adjusts the model so the guesses improve.
```

PR #7 mostly adds these core files:

```text
apps/cv/models/__init__.py
apps/cv/models/recognizer.py
apps/cv/training/train_recognizer.py
apps/cv/tests/test_plate_recognizer.py
```

It also includes review/deployment fixes in files like:

```text
Dockerfile
docker-compose.yml
config/settings.py
config/urls.py
pytest.ini
```

Those review-fix files are important for running the project safely, but the
machine learning learning path is mainly the model, training script, dataset
helpers, and tests.

## First: The Whole Recognizer In One Simple Story

Imagine the input plate says:

```text
ABC123
```

The recognizer does not see it as text. It sees a small grayscale image:

```text
shape: (B, 1, 32, 128)
```

For one image, you can think:

```text
B = 1 image
1 = grayscale channel
32 = pixels tall
128 = pixels wide
```

Then the model does this:

```text
1. CNN looks for visual shapes.
2. CNN shrinks the image width from 128 to 16.
3. Each of those 16 width positions becomes one sequence step.
4. LSTM reads those 16 steps like a sentence.
5. Linear layer turns each step into 37 scores.
6. log_softmax turns scores into log-probabilities.
7. CTC decoding cleans the 16 guesses into final text.
```

Example of messy CTC output:

```text
A A blank B blank C C blank 1 blank 2 2 blank 3 blank blank
```

After CTC cleanup:

```text
ABC123
```

## File: `apps/cv/models/__init__.py`

This file is tiny, but it makes the model package easier to use.

Current file:

```python
from apps.cv.models.plate_detector import PlateDetectorCNN
from apps.cv.models.recognizer import PlateRecognizerCRNN

__all__ = ["PlateDetectorCNN", "PlateRecognizerCRNN"]
```

Line-by-line:

```text
Line 11 imports PlateDetectorCNN.
```

That is the older model that finds the location of a plate in a full image.

```text
Line 12 imports PlateRecognizerCRNN.
```

That is the new PR #7 model that reads text from a cropped plate image.

```text
Line 14 defines __all__.
```

`__all__` tells Python which names this package officially exports.

So instead of writing:

```python
from apps.cv.models.plate_detector import PlateDetectorCNN
from apps.cv.models.recognizer import PlateRecognizerCRNN
```

other code can write:

```python
from apps.cv.models import PlateDetectorCNN, PlateRecognizerCRNN
```

That is cleaner.

## File: `apps/cv/models/recognizer.py`

This is the actual neural network.

The file creates:

```python
class PlateRecognizerCRNN(nn.Module):
```

`CRNN` means:

```text
C = Convolutional
R = Recurrent
NN = Neural Network
```

Plain English:

```text
CNN part: looks at the picture.
LSTM part: reads the picture features in order.
```

### Lines 1-37: The Big Docstring

The first big triple-quoted block is a docstring. It explains the model before
the code starts.

Important parts:

```text
Input:  (B, 1, 32, 128)
Output: (T=16, N, C=37)
```

Think of this like:

```text
Input:  a batch of tiny plate pictures
Output: 16 left-to-right guesses per picture
```

Why 37 classes?

```text
1 blank token
26 letters A-Z
10 digits 0-9
```

```text
1 + 26 + 10 = 37
```

The blank token is special. It means:

```text
"I do not think there is a character at this time step."
```

### Lines 39-43: Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

These import PyTorch.

`torch` is the main library.

`torch.nn` contains neural network building blocks like:

```text
Conv2d
BatchNorm2d
ReLU
MaxPool2d
LSTM
Linear
```

`torch.nn.functional` contains functions like:

```text
log_softmax
```

Then:

```python
from apps.cv.training.dataset import BLANK_IDX, IDX_TO_CHAR, VOCAB_SIZE
```

This imports the vocabulary rules:

```text
BLANK_IDX = 0
VOCAB_SIZE = 37
IDX_TO_CHAR maps numbers back to letters/digits
```

Example:

```text
1 -> A
2 -> B
...
26 -> Z
27 -> 0
28 -> 1
```

### Lines 46-66: Class Definition

```python
class PlateRecognizerCRNN(nn.Module):
```

This creates a PyTorch model class.

In PyTorch, every model usually inherits from `nn.Module`.

That gives the model useful behavior:

```text
model.parameters()
model.train()
model.eval()
model.to(device)
model(x)
```

The class docstring says what the model expects:

```text
Input shape:  (B, 1, 32, 128)
Output shape: (T=16, N, C=37)
```

`B` and `N` both mean batch size in this file. PyTorch CTC documentation often
uses `N`, while image code often uses `B`.

### Lines 68-70: Important Constants

```python
_DROPOUT: float = 0.3
_SEQUENCE_LEN: int = 16
_LSTM_INPUT: int = 2048
```

These are model settings.

`_DROPOUT = 0.3` means:

```text
During training, randomly hide 30% of some internal signals.
```

Why hide signals? To stop the model from memorizing too much. It forces the
model to learn stronger patterns.

`_SEQUENCE_LEN = 16` means:

```text
After the CNN, the image width becomes 16 positions.
The LSTM reads 16 steps.
```

`_LSTM_INPUT = 2048` comes from:

```text
256 channels * 8 height rows = 2048 features
```

After the CNN, each vertical slice has:

```text
256 learned feature maps
8 rows tall
```

Flatten those together:

```text
256 * 8 = 2048
```

### Lines 72-73: Constructor Start

```python
def __init__(self, dropout: float = _DROPOUT) -> None:
    super().__init__()
```

`__init__` builds the model layers.

`super().__init__()` calls the PyTorch parent setup. Without this, PyTorch would
not correctly track the layers and parameters.

### Lines 90-95: CNN Block 1

```python
self.block1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
```

This is the first image-processing block.

`nn.Sequential(...)` means:

```text
Run these layers in order.
```

Layer 1:

```python
nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
```

Beginner explanation:

```text
Look at tiny 3x3 patches of the image and learn 64 different visual detectors.
```

Why `1` input channel?

```text
The plate image is grayscale, so it has one channel.
```

Why `64` output channels?

```text
The model learns 64 kinds of simple patterns.
Examples: edges, corners, dark strokes, bright strokes.
```

Why `kernel_size=3`?

```text
The filter looks at 3x3 pixel neighborhoods.
```

Why `padding=1`?

```text
It keeps the height and width the same during convolution.
```

Without padding, the image would shrink every time a convolution runs.

Layer 2:

```python
nn.BatchNorm2d(64)
```

This keeps the numbers stable during training. A beginner way to think about it:

```text
It stops the layer outputs from becoming too wild.
```

Layer 3:

```python
nn.ReLU(inplace=True)
```

ReLU means:

```text
negative numbers become 0
positive numbers stay
```

Example:

```text
[-2, -0.5, 0, 3] -> [0, 0, 0, 3]
```

Why do this? It adds non-linearity, which lets the model learn complicated
patterns instead of only simple straight-line math.

Layer 4:

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

This shrinks the image by taking the strongest signal in each 2x2 area.

Shape change:

```text
(B, 1, 32, 128)
-> (B, 64, 16, 64)
```

Plain English:

```text
The image becomes half as tall and half as wide.
The model now has 64 learned feature channels.
```

### Lines 100-105: CNN Block 2

```python
self.block2 = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
```

This is similar to block 1, but deeper.

Input channels:

```text
64
```

Output channels:

```text
128
```

Shape change:

```text
(B, 64, 16, 64)
-> (B, 128, 8, 32)
```

Beginner meaning:

```text
Block 1 learned simple edges.
Block 2 combines edges into bigger shapes.
```

Examples:

```text
vertical stroke
round curve
diagonal line
partial letter shape
```

### Lines 117-122: CNN Block 3

```python
self.block3 = nn.Sequential(
    nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
)
```

This creates the highest-level image features.

Shape change:

```text
(B, 128, 8, 32)
-> (B, 256, 8, 16)
```

Notice the height stays 8.

Why?

```python
MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
```

This means:

```text
height pooling = 1, so height does not shrink
width pooling = 2, so width shrinks from 32 to 16
```

This is important for license plates. Characters like these can be confused:

```text
I and 1
O and 0
B and 8
```

Keeping height detail gives the model more vertical information.

### Lines 143-150: The LSTM

```python
self.lstm = nn.LSTM(
    input_size=self._LSTM_INPUT,
    hidden_size=256,
    num_layers=2,
    bidirectional=True,
    dropout=dropout,
    batch_first=False,
)
```

This is the sequence-reading part.

Before this, the CNN created:

```text
(B, 256, 8, 16)
```

The width `16` becomes 16 time steps:

```text
step 1, step 2, step 3, ... step 16
```

`input_size=2048` means each step has 2048 numbers.

`hidden_size=256` means each LSTM direction creates 256 features per step.

`num_layers=2` means two LSTM layers are stacked.

`bidirectional=True` means it reads:

```text
left -> right
right -> left
```

This helps because context matters.

Example:

```text
If a character looks like O or 0, nearby characters can help.
```

If the rest of the plate pattern looks like:

```text
ABC?23
```

the model may learn that the `?` is more likely to be `1` or a digit in some
plate formats, depending on the synthetic data.

`batch_first=False` means the LSTM expects:

```text
(time, batch, features)
```

not:

```text
(batch, time, features)
```

That is why the `forward()` method later uses `permute`.

### Line 157: Final Linear Layer

```python
self.fc = nn.Linear(512, VOCAB_SIZE)
```

The LSTM outputs 512 features per time step because:

```text
256 forward features + 256 backward features = 512
```

The final layer turns those 512 features into 37 scores.

So for each of the 16 time steps, the model asks:

```text
How likely is blank?
How likely is A?
How likely is B?
...
How likely is Z?
How likely is 0?
...
How likely is 9?
```

### Lines 159-199: `forward()`

`forward()` is the most important method in a PyTorch model.

When you write:

```python
out = model(x)
```

PyTorch actually calls:

```python
model.forward(x)
```

Line 176:

```python
B = x.size(0)
```

This reads the batch size.

Example:

```text
x shape = (4, 1, 32, 128)
B = 4
```

Lines 179-181:

```python
x = self.block1(x)
x = self.block2(x)
x = self.block3(x)
```

The image passes through the CNN blocks.

Shape story:

```text
(B, 1, 32, 128)
-> (B, 64, 16, 64)
-> (B, 128, 8, 32)
-> (B, 256, 8, 16)
```

Line 191:

```python
x = x.reshape(B, self._LSTM_INPUT, self._SEQUENCE_LEN)
```

This changes:

```text
(B, 256, 8, 16)
```

into:

```text
(B, 2048, 16)
```

because:

```text
256 * 8 = 2048
```

Line 192:

```python
x = x.permute(2, 0, 1)
```

This reorders the dimensions.

Before:

```text
(B, 2048, 16)
```

After:

```text
(16, B, 2048)
```

Why?

Because the LSTM wants:

```text
(time, batch, features)
```

Line 195:

```python
x, _ = self.lstm(x)
```

The LSTM reads the 16 time steps.

Input:

```text
(16, B, 2048)
```

Output:

```text
(16, B, 512)
```

Line 198:

```python
x = self.fc(x)
```

The model converts 512 LSTM features into 37 class scores.

Shape:

```text
(16, B, 512)
-> (16, B, 37)
```

Line 199:

```python
return F.log_softmax(x, dim=-1)
```

This turns raw scores into log-probabilities.

`dim=-1` means:

```text
apply it across the 37 character classes
```

This is correct because each time step should choose from the 37 classes.

Do not apply `log_softmax` again later. The model already does it.

### Lines 201-226: `predict()`

```python
@torch.no_grad()
def predict(self, x: torch.Tensor) -> torch.Tensor:
```

`@torch.no_grad()` means:

```text
Do not track gradients.
```

Gradients are needed for training, but not for prediction. Turning them off
makes prediction faster and uses less memory.

Line 220:

```python
was_training = self.training
```

This remembers whether the model was in training mode.

Line 221:

```python
self.eval()
```

This switches the model to evaluation mode.

Why? Dropout should be off during prediction. If dropout stayed on, the same
image could give slightly different predictions.

Lines 222-226:

```python
try:
    return self.forward(x)
finally:
    if was_training:
        self.train()
```

This means:

```text
Run prediction.
Then restore training mode if the model was training before.
```

That is careful engineering. It prevents `predict()` from accidentally breaking
a training loop.

### Lines 228-269: `decode_predictions()`

This method turns the model output into readable plate strings.

Input shape:

```text
(T, N, C)
```

Example:

```text
T = 16 time steps
N = 2 images in the batch
C = 37 classes
```

Line 248:

```python
indices = output.argmax(dim=-1)
```

`argmax` means:

```text
pick the biggest value
```

So at each time step, the model picks its favorite class.

Example:

```text
time 1: A has biggest score -> choose A
time 2: A has biggest score -> choose A
time 3: blank has biggest score -> choose blank
time 4: B has biggest score -> choose B
```

Lines 250-252:

```python
decoded = []
for n in range(indices.size(1)):
    seq = indices[:, n].tolist()
```

This loops through each image in the batch.

If `N = 4`, it decodes 4 plates.

Lines 258-261:

```python
collapsed = []
for token in seq:
    if not collapsed or token != collapsed[-1]:
        collapsed.append(token)
```

This collapses repeated neighboring tokens.

Example:

```text
[A, A, A, B, B, blank, C, C]
```

becomes:

```text
[A, B, blank, C]
```

Why? In CTC, the model may repeat a character for several frames to show that
the character lasts across part of the image.

Lines 266-267:

```python
chars = [IDX_TO_CHAR[tok] for tok in collapsed if tok != BLANK_IDX]
decoded.append("".join(chars))
```

This removes blank tokens and converts numbers back into characters.

Example:

```text
[1, 2, blank, 3]
```

becomes:

```text
["A", "B", "C"]
```

then:

```text
"ABC"
```

## File: `apps/cv/training/train_recognizer.py`

This file trains the recognizer model.

Training means:

```text
show the model many examples
make it guess
measure how wrong it is
adjust the weights
repeat many times
```

### Lines 1-34: Script Docstring

The top of the file explains how to run it.

Important command:

```bash
python apps/cv/training/train_recognizer.py \
    --data-dir data/recognizer \
    --epochs 100 \
    --output apps/cv/weights/recognizer.pth
```

This means:

```text
Use training data from data/recognizer.
Train for 100 epochs.
Save best learned weights to apps/cv/weights/recognizer.pth.
```

### Lines 36-55: Imports And Path Fix

The script imports Python tools:

```python
import argparse
import logging
import sys
from pathlib import Path
```

`argparse` reads command-line options.

`logging` prints useful messages.

`sys` lets the script adjust Python import paths.

`Path` is a clean way to work with file paths.

Then:

```python
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

This fixes a real PR review issue.

When you run:

```bash
python apps/cv/training/train_recognizer.py
```

Python normally puts this folder on the import path:

```text
apps/cv/training
```

But the project imports need the repository root so imports like this work:

```python
from apps.cv.models.recognizer import PlateRecognizerCRNN
```

So the script adds the repo root before importing project modules.

### `_train_epoch()`: One Full Training Pass

`_train_epoch()` trains through the training dataset once.

Simple meaning:

```text
For each batch:
  move images to the device
  run the model
  calculate loss
  backpropagate
  clip gradients
  update model weights
```

Important lines:

```python
model.train()
```

This turns training behavior on. Dropout is active.

```python
images = batch["images"].to(device)
```

Images move to the selected device:

```text
MPS, CUDA, or CPU
```

```python
targets = batch["targets"]
target_lengths = batch["target_lengths"]
```

These stay on CPU because CTCLoss is run on CPU.

Example:

```text
Plate 1 label: ABC123
Plate 2 label: Z9
```

The collate function stores targets like:

```text
[A, B, C, 1, 2, 3, Z, 9]
```

and lengths like:

```text
[6, 2]
```

That tells CTCLoss how to split the target list back into each plate.

```python
optimizer.zero_grad()
```

Before calculating new gradients, clear old gradients.

If you forget this, gradients pile up from previous batches.

```python
log_probs = model(images)
```

This runs the recognizer.

Output:

```text
(T=16, N, C=37)
```

```python
input_lengths = torch.full((N,), T, dtype=torch.long)
```

Every image has 16 time steps, so this creates:

```text
[16, 16, 16, ...]
```

one per image in the batch.

```python
loss = criterion(log_probs.cpu(), targets, input_lengths, target_lengths)
```

This calculates CTC loss.

Loss means:

```text
How wrong was the model?
```

Lower loss is better.

The `.cpu()` is important because PyTorch MPS does not support CTCLoss.

```python
loss.backward()
```

This is backpropagation.

It calculates:

```text
Which weights contributed to the mistake?
How should each weight change?
```

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

This prevents gradients from becoming too huge.

Huge gradients can make training unstable, like taking a step that is too large
and jumping past the solution.

```python
optimizer.step()
```

This updates the weights.

That is the moment learning actually happens.

### `_validate_epoch()`: Checking Progress Without Learning

Validation means:

```text
Test the model on examples it is not training on.
Do not update weights.
Just measure how well it does.
```

The function starts with:

```python
@torch.no_grad()
```

and:

```python
model.eval()
```

So validation is stable and does not track gradients.

It calculates three things:

```text
validation loss
character accuracy
plate accuracy
```

Character accuracy example:

```text
prediction: ABC12X
truth:      ABC123
```

5 out of 6 characters match:

```text
83.3% character accuracy
```

Plate accuracy is stricter:

```text
prediction must exactly equal truth
```

So in that same example:

```text
plate accuracy = 0% for that plate
```

because one character is wrong.

### `_smooth()`: Making Noisy Loss Easier To See

Training loss can bounce around batch to batch.

`_smooth()` applies an exponential moving average.

Simple example:

```text
raw losses:      5.0, 3.0, 4.0, 2.0
smoothed losses: less jumpy version of those numbers
```

This is only for the chart. It does not change training.

### `_plot_training_history()`: Drawing The Training Chart

This function creates a PNG chart with:

```text
train loss
validation loss
character accuracy
plate accuracy
learning rate
```

Why is this useful?

Because a chart helps you see training behavior.

Good training usually looks like:

```text
loss goes down
accuracy goes up
```

Bad signs:

```text
train loss goes down but validation loss goes up
validation accuracy never improves
learning rate drops too often
```

### `_parse_args()`: Reading Command-Line Options

This function defines options like:

```text
--data-dir
--epochs
--batch-size
--lr
--output
--seed
```

Example:

```bash
python apps/cv/training/train_recognizer.py --epochs 2 --batch-size 4
```

Then Python stores those values in `args`.

### `main()`: The Script's Main Plan

`main()` runs the whole training workflow.

Step 1:

```python
args = _parse_args()
```

Read command-line options.

Step 2:

```python
torch.manual_seed(args.seed)
```

Set the random seed so train/validation splitting is reproducible.

Step 3:

```python
device = get_device()
```

Pick the best available device:

```text
MPS on Apple Silicon
CUDA on NVIDIA GPU
CPU otherwise
```

Step 4:

```python
dataset = PlateRecognizerDataset(args.data_dir)
```

Load the recognizer dataset.

Step 5:

```python
train_set, val_set = random_split(dataset, [0.8, 0.2], generator=rng)
```

Split data:

```text
80% training
20% validation
```

Step 6:

```python
DataLoader(..., collate_fn=ctc_collate_fn)
```

Build loaders that create batches.

`ctc_collate_fn` is required because plate labels have different lengths.

Example:

```text
ABC123 has 6 characters
AB12345 has 7 characters
Z9 has 2 characters
```

Normal batching wants equal-length labels. CTC batching handles variable-length
labels by concatenating them and storing lengths.

Step 7:

```python
model = PlateRecognizerCRNN().to(device)
criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)
```

This creates:

```text
model: the neural network
criterion: the loss function
optimizer: the weight updater
scheduler: the learning-rate adjuster
```

`zero_infinity=True` prevents infinite CTC losses from breaking training early.

Step 8:

```python
if not output_path.is_relative_to(REPO_ROOT):
    raise SystemExit(...)
```

This is a safety check. It prevents someone from saving model output outside the
project folder using a path like:

```text
../../../../somewhere-dangerous/file.pth
```

Step 9:

```python
for epoch in range(1, args.epochs + 1):
```

Train for the requested number of epochs.

Inside each epoch:

```text
train once
validate once
update learning rate
record history
save best weights if validation improved
```

Step 10:

```python
torch.save(model.state_dict(), output_path)
```

This saves the learned weights.

`state_dict()` is safer and cleaner than saving the whole Python object.

Step 11:

```python
_plot_training_history(history, output_path, best_epoch)
```

Save the training chart.

## File: `apps/cv/tests/test_plate_recognizer.py`

Tests are not part of the model's learning, but they protect the code.

They answer questions like:

```text
Does the model output the right shape?
Does predict() turn gradients off?
Does decoding handle CTC blanks correctly?
Does the script run as documented?
```

### Lines 13-21: Test Imports

```python
import subprocess
import sys
from pathlib import Path

import pytest
import torch
```

These imports support:

```text
running subprocess commands
checking Python executable paths
creating test tensors
marking pytest tests
```

```python
from apps.cv.models.recognizer import PlateRecognizerCRNN
from apps.cv.training.dataset import BLANK_IDX, VOCAB_SIZE
```

The tests import the model and shared vocabulary constants.

### Lines 26-28: Random Plate Helper

```python
def _random_plate_batch(batch_size: int = 2) -> torch.Tensor:
    return torch.rand(batch_size, 1, 32, 128)
```

This creates fake plate images.

They are random noise, not real plates, but that is fine for shape tests.

Example:

```text
batch_size = 4
shape = (4, 1, 32, 128)
```

The tests do not need real images because they are checking whether the model's
math pipeline is wired correctly.

### Lines 37-42: Output Shape Test

```python
model = PlateRecognizerCRNN()
x = _random_plate_batch(batch_size=4)
out = model(x)
assert out.shape == (16, 4, VOCAB_SIZE)
```

This proves:

```text
4 input images produce 4 output sequences.
Each sequence has 16 time steps.
Each time step has 37 class scores.
```

### Lines 44-49: Single Image Test

This checks batch size 1.

Why important?

Sometimes code accidentally removes a dimension when batch size is 1. This test
protects against that.

Expected:

```text
(16, 1, 37)
```

### Lines 51-66: Log Softmax Test

```python
assert out.max().item() <= 0.0
```

Log probabilities are always less than or equal to zero.

Why?

Probability values are between 0 and 1:

```text
0.1, 0.5, 1.0
```

Logs of those are:

```text
log(0.1) = negative
log(0.5) = negative
log(1.0) = 0
```

So if the output has a positive number, it is not a valid log probability.

### Lines 68-82: Probability Sum Test

```python
class_probs = out.exp()
row_sums = class_probs.sum(dim=-1)
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
```

`out` is log-probabilities.

`out.exp()` turns them back into normal probabilities.

For each time step, all 37 class probabilities should add up to 1.

Example:

```text
blank: 0.10
A:     0.60
B:     0.20
...
all classes together: 1.00
```

### Lines 103-134: `predict()` Tests

These tests check that `predict()`:

```text
returns the same output shape as forward()
does not track gradients
restores train mode if the model was training
keeps eval mode if the model was already evaluating
```

This matters because prediction should be safe to call anywhere.

### Lines 136-170: Dropout Behavior Tests

In eval mode:

```text
same input -> same output
```

In train mode:

```text
same input -> slightly different output
```

Why? Dropout randomly hides signals during training.

These tests make sure the training/eval modes actually behave differently.

### Lines 172-184: Parameter Count Test

```python
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert 5_000_000 <= total <= 20_000_000
```

This counts trainable weights.

Why test this?

Because if someone accidentally deletes the LSTM or changes a layer size, the
parameter count will change a lot.

This test catches major architecture mistakes.

### Lines 188-271: Decoder Tests

These tests protect CTC decoding.

All blank:

```text
[blank, blank, blank] -> ""
```

Repeated characters without blanks:

```text
[A, A, A, B, B] -> "AB"
```

Repeated characters with blanks:

```text
[A, blank, A] -> "AA"
```

That last case is very important. The blank separates repeated real characters.

### Lines 274-299: Documented Script Command Test

```python
result = subprocess.run(
    [
        sys.executable,
        "apps/cv/training/train_recognizer.py",
        "--help",
    ],
    cwd=repo_root,
    capture_output=True,
    text=True,
    check=False,
)
```

This runs the training script the same way the docs tell users to run it.

It does not train the model. It only runs:

```text
--help
```

That is enough to prove the script can start and import project modules.

The final checks:

```python
assert result.returncode == 0, result.stderr
assert "--data-dir" in result.stdout
```

Mean:

```text
the command did not crash
the help text mentions --data-dir
```

## Dataset Helpers Used By PR #7

PR #7 depends on shared recognizer dataset helpers in:

```text
apps/cv/training/dataset.py
```

Important constants:

```python
BLANK_IDX = 0
_CHARS = string.ascii_uppercase + string.digits
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(_CHARS)}
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(_CHARS) + 1
```

Plain English:

```text
blank -> 0
A -> 1
B -> 2
...
Z -> 26
0 -> 27
1 -> 28
...
9 -> 36
```

Why not make `A` equal 0?

Because CTC reserves 0 for blank.

### `ctc_collate_fn`

Normal DataLoader batching works best when every label has the same length.

But license plates can have different lengths:

```text
ABC123   length 6
AB12345  length 7
Z9       length 2
```

`ctc_collate_fn` solves this by producing:

```text
images: stacked image tensor
targets: one long list of all label indices
target_lengths: how long each original label was
```

Example:

```text
Plate 1: ABC
Plate 2: 12
```

Encoded:

```text
ABC -> [1, 2, 3]
12  -> [28, 29]
```

Collated:

```text
targets = [1, 2, 3, 28, 29]
target_lengths = [3, 2]
```

CTCLoss uses `target_lengths` to know:

```text
first 3 tokens belong to plate 1
next 2 tokens belong to plate 2
```

## Review Fixes Included Around PR #7

PR #7 also received review feedback and fixes.

### Training Script Import Fix

Problem:

```text
python apps/cv/training/train_recognizer.py
```

could fail because Python did not know where the project root was.

Fix:

```python
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

Result:

```text
The documented command works without manually setting PYTHONPATH.
```

### Docker Dev Dependency Fix

Problem:

```text
DEBUG=True adds django_extensions to INSTALLED_APPS.
But docker-compose originally installed only requirements.txt.
django_extensions lives in requirements-dev.txt.
```

That could make Docker startup fail.

Fix:

```text
docker-compose passes INSTALL_DEV_REQUIREMENTS=true.
Dockerfile installs requirements-dev.txt for the local dev image.
```

### Proxy HTTPS Fix

Problem:

```text
If nginx/load balancer handles HTTPS, Django may only see HTTP.
SECURE_SSL_REDIRECT could redirect forever.
```

Fix:

```python
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
```

This tells Django:

```text
If the proxy says the original request was HTTPS, treat it as secure.
```

### Pytest Coverage Fix

Problem:

```text
pytest apps/cv/tests/test_plate_recognizer.py -v
```

should only run CV tests.

But global coverage settings were forcing accounts/parking coverage checks on
every pytest command.

Fix:

```text
pytest.ini keeps only --tb=short globally.
Coverage gate stays in the explicit documented command.
```

### Health Check Fix

Problem:

```python
connection.ensure_connection()
```

can say a connection object exists without proving the database can answer.

Fix:

```python
with connection.cursor() as cursor:
    cursor.execute('SELECT 1')
    cursor.fetchone()
```

Now the health check asks the database a tiny real question.

## If You Only Remember Five Things

1. The recognizer reads cropped plate images, not full parking-lot images.

2. The CNN turns the image into learned visual features.

3. The LSTM reads those features from left to right and right to left.

4. CTC lets the model learn plate text without knowing the exact pixel location
   of each character.

5. The training script repeats this loop:

```text
guess -> measure loss -> backpropagate -> update weights -> validate -> save best
```

That is the heart of this PR.
