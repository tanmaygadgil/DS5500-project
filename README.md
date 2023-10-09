# DS5500-project

## Dataset

- [https://physionet.org/content/grabmyo/1.0.2/](https://physionet.org/content/grabmyo/1.0.2/)

## Summary

The advent of smart devices capable of interpreting and responding to gestures has ushered in a
new era of accessibility and independence for those facing mobility challenges. By harnessing the
power of gestures, we envision a world where people with disabilities can control their environments,
communicate effectively, navigate mobility aids, and engage in activities that were once limited or
inaccessible. State-of-the-art devices such as gesture-controlled wheelchairs, home automation devices,
and prosthetic arms/limbs can be extremely helpful in navigating the day-to-day for the disabled
population.
Verifying and authenticating the use of gesture-controlled devices for individuals with mobility
issues is paramount to ensure their accessibility, effectiveness, and safety. This involves verifying and
identifying gestures, user recognition, and feedback collection from the devices.
One way to enable gesture control in these devices is to make use of EMG (electromyography)
signals. EMG (electromyography) can be used in wearable devices to detect muscle activation and
interpret movement intent. By placing EMG sensors on the skin over targeted muscles, small electrical
signals generated during muscle contractions can be measured. This allows wearables using EMG to
track gestures, finger movements, and other precise motions to control devices or provide biofeedback
on muscle activity.
Our goal is to use EMG data from different sensor suites and build out a family of models that can
perform tasks like gesture recognition and user authentication. This effort will also explore techniques
to efficiently ingest and process (pre-processing and post-processing) EMG data for different sensor
suites.

## Proposed plan

We aim to explore the possible applications of this data in two phases. The first phase will focus on
building a reliable classifier for different hand gestures. The second phase will focus on building an
authentication system.
The first phase will be subdivided into several tasks like feature engineering, modeling, and model
analysis. To extract features from the EMG data, we will attempt to use traditional signal processing
techniques[3]. We will first segment the data into chunks that can be ingested by our learning
algorithms. We can also extract several waveform statistics as features such as the root mean square
(RMS), variance (VAR), mean absolute value (MAV), slope sign change (SSC), zero crossing (ZC),
and waveform length (WL).
The modeling effort will mainly comprise building traditional machine learning classification models
like Support Vector Machines, Logistics Regression, Gradient-boosted Trees, and different architectures
of the multi-layer perception. As a stretch goal, we intend to experiment with complex time-based
Neural Network models such as LSTMs or traditional signal processing classifiers. Our model analysis
effort will comprise using explainability tools like SHAPley values and model-based explainability
methods like feature importance and coefficient analysis.


Some Links:
- [https://github.com/SebastianRestrepoA/EMG-pattern-recognition](https://github.com/SebastianRestrepoA/EMG-pattern-recognition): We have used these code snippets as an inspiration for our EDA.
