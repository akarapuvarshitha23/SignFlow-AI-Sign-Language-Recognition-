# 🤟 SignFlow AI - Real-Time Sign Language Recognition & Translation Web Application

A full-stack Deep Learning and Computer Vision web application developed to bridge the communication gap between individuals with hearing/speech impairments and the wider community.

Captures hand gestures through a standard webcam, recognizes gestures using a trained Deep Learning model with **MediaPipe 3D Hand Landmarks**, converts them into continuous text with intelligent debounce filtering, and generates natural vocalized speech using the **Web Speech API**.

---

## 🌟 Key Features

### A. Real-Time Sign Language Detection
- **Webcam Access**: Seamless browser-based webcam integration with Start Camera and Stop Camera controls.
- **3D Landmark Tracking**: MediaPipe Hand Landmarker extracts 21 3D joints $(x, y, z)$ in real time.
- **Deep Learning Classifier**: Multi-layer Perceptron (MLP) with Batch Normalization & Dropout regularizations trained to **100.00% test accuracy**.
- **Real-Time Visual Overlay**: Live skeletal bones, glowing joints, and lavender/blue bounding box rendered directly on screen.
- **Live Confidence & Latency**: Shows primary prediction badge, percentage confidence meter, and top-3 probability distribution bars with ~0.58 ms inference latency.
- **Graceful Error Handling**: Displays informative notices when camera permission is requested or when no hand is in frame.

### B. Gesture-to-Text Conversion
- **Continuous Sentence Accumulator**: Automatically combines recognized gestures into coherent sentences.
- **Hysteresis & Anti-Flicker Debounce**: Prevents repeating the same character across continuous 30 FPS video frames with a 1.2s cooldown timer.
- **Sentence Controls**: Includes dedicated buttons for **Clear Text**, **Add Space**, and **Delete Last Character** (Backspace).

### C. Text-to-Speech (TTS) Conversion
- **Vocalized Communication**: Native Web Speech API integration speaks recognized sentences aloud.
- **Audio Controls**: **Speak**, **Pause**, **Resume**, and **Stop** buttons with a real-time animated audio waveform visualizer.
- **Voice Customization**: Select system voices, speech rate (speed 0.5x - 2.0x), and pitch.

### D. Learn Sign Language & Text-to-Sign
- **Interactive ASL Dictionary**: Visual cards for **Alphabets (A–Z)**, **Numbers (0–9)**, and **Common Phrases** with anatomical tips and gesture descriptions.
- **Interactive Text-to-Sign Translator**: Type any text or numbers (e.g. `HELLO 123`) and instantly see the animated sequential sign illustrations rendered in real-time.
- **Step-Through Animation**: "Animate Sequence" button sequentially highlights each sign card for educational practice.

### E. 5-Page Responsive Web Interface
1. **Home Page**: Project hero section, value proposition, live metrics counter, and 4-step pipeline explanation.
2. **Live Detection Page**: Video canvas, start/stop buttons, real-time prediction, sentence translator, audio controls.
3. **Text-to-Speech Page**: Dedicated speech lab with multi-line text input, voice selectors, rate/pitch sliders, and quick phrase presets.
4. **Learn Sign Language Page**: Visual sign dictionary with search filters and interactive Text-to-Sign converter.
5. **About Project Page**: Problem statement, mathematical formulation of invariant feature engineering, neural network topology, and benchmark report.

---

## 🗂️ Project Directory Structure

```text
├── backend/
│   ├── __init__.py               # Backend package initializers
│   ├── landmark_extractor.py     # MediaPipe 3D tracking & 76D feature extraction
│   ├── model_builder.py          # Keras Deep Neural Network definition
│   ├── sign_detector.py          # Real-time detector, debounce, & sentence logic
│   └── trainer.py                # Dataset loader, model training, & evaluation
│
├── frontend/
│   ├── package.json              # Optional standalone React app configuration
│   └── src/
│       └── App.jsx               # Modular React single page application source
│
├── model/
│   ├── hand_landmarker.task      # MediaPipe 3D Hand Landmarker asset
│   ├── sign_model.keras          # Trained Deep Neural Network classifier
│   ├── label_map.json            # Class index to sign name mapping
│   ├── model_metadata.json       # Training stats and evaluation metrics
│   └── training_curves.png       # Training/validation accuracy & loss plots
│
├── dataset/
│   ├── dataset_generator.py      # Synthesizes 3D hand poses & captures live gestures
│   └── sign_data.csv             # 8,550 hand landmark feature vectors
│
├── static/
│   ├── css/
│   │   └── style.css             # Premium soft ivory, lavender & sky blue theme
│   ├── js/
│   │   ├── app.js                # Frontend SPA controller & Web Speech API TTS
│   │   └── sign_data.js          # SVG sign illustrations & anatomical descriptions
│   └── images/signs/             # Sign icon assets
│
├── templates/
│   └── index.html                # Responsive 5-page Single Page Architecture
│
├── app.py                        # Flask Web Server & REST API
├── run_live.py                   # Standalone desktop OpenCV HUD application
├── demo_test.py                  # Automated benchmark & accuracy verification test
├── requirements.txt              # Project dependencies
└── README.md                     # Comprehensive documentation
```

---

## 🧠 Deep Learning Architecture & Mathematical Pipeline

### Invariant 76-Dimensional Feature Engineering
Raw $(x, y, z)$ coordinates change whenever the user sits closer or further from the camera. SignFlow AI ensures 100% position and distance invariance:

1. **Translation Invariance**:
   $$\mathbf{P}_i' = \mathbf{P}_i - \mathbf{P}_0 \quad \text{for } i \in \{0, \dots, 20\}$$
   Where $\mathbf{P}_0$ is the wrist landmark. The wrist is always placed at origin $(0, 0, 0)$.

2. **Scale Invariance**:
   $$d_{\max} = \max_{i} \|\mathbf{P}_i'\|_2$$
   $$\mathbf{P}_i'' = \frac{\mathbf{P}_i'}{d_{\max}}$$
   Normalizes the hand to unit radius regardless of distance to webcam.

3. **Engineered Geometric Distances**:
   - 63 normalized 3D coordinates ($21 \times 3$).
   - 5 wrist-to-fingertip Euclidean distances (Thumb, Index, Middle, Ring, Pinky).
   - 4 adjacent fingertip Euclidean spans (Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky).
   - 4 thumb-to-fingertip Euclidean distances (crucial for pinching signs like OK/F, O, D).
   - **Total Feature Vector Dimension: 76 Features**.

### Deep Neural Network Topology
```text
Input (76 Features)
  │
  ├── Dense (256, L2=1e-4) ──▶ BatchNorm ──▶ ReLU ──▶ Dropout (0.30)
  ├── Dense (128, L2=1e-4) ──▶ BatchNorm ──▶ ReLU ──▶ Dropout (0.25)
  ├── Dense (64,  L2=1e-4) ──▶ BatchNorm ──▶ ReLU ──▶ Dropout (0.20)
  └── Dense (19 Classes)   ──▶ Softmax Activation
```

---

## 📊 Evaluation & Verification Results

Evaluated on 1,283 held-out test samples and 950 validation trials:

- **Overall Test Accuracy**: **100.00%**
- **Test Loss**: **0.0203**
- **Single Sample Latency**: **~0.58 ms** (>1,700 FPS on CPU)
- **Model Parameters**: 63,891 parameters (Lightweight ~249 KB)

---

## 🚀 Execution & Usage Guide

### 1. Run the Web Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

- Allow browser camera permissions when prompted.
- Click **"Start Camera"** on the Live Detection page.
- Perform hand gestures in front of the webcam.
- Click **"Speak Sentence"** to hear the translated words.
- Explore the **"Learn Sign Language"** tab to practice signs and type words in the **Text-to-Sign converter**.

### 2. Standalone Desktop OpenCV Application
If you want to run a native desktop window with a custom HUD:
```bash
python run_live.py
```
- Press `C` to clear sentence.
- Press `SPACE` to add a space.
- Press `BACKSPACE` to delete last character.
- Press `Q` or `ESC` to quit.

### 3. Automated Benchmark Test
To benchmark the model on test samples across all classes:
```bash
python demo_test.py
```

### 4. Retrain the Deep Learning Model
To retrain the model and regenerate evaluation curves:
```bash
python backend/trainer.py
```
To capture custom live gestures with your webcam:
```bash
python dataset/dataset_generator.py --record MY_GESTURE --samples 150
```
