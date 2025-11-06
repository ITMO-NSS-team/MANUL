import numpy as np
from matplotlib import pyplot as plt

# station number/channel/frequency/user/batch
def extract_geometric_features(complex_channel_data):
    """Convert complex channel to geometric features"""
    features = []

    # Amplitude/phase features
    features.append(np.abs(complex_channel_data))  # Magnitude
    features.append(np.angle(complex_channel_data))  # Phase
    features.append(np.real(complex_channel_data))  # I component
    features.append(np.imag(complex_channel_data))  # Q component

    # Statistical features across dimensions
    if complex_channel_data.ndim > 1:
        features.append(np.mean(np.abs(complex_channel_data), axis=0))  # Mean magnitude
        features.append(np.std(np.abs(complex_channel_data), axis=0))  # Magnitude variation
        features.append(np.mean(np.angle(complex_channel_data), axis=0))  # Mean phase
        features.append(np.unwrap(np.angle(complex_channel_data), axis=0))  # Unwrapped phase

    # Power delay profile (for time-domain channels)
    if complex_channel_data.ndim > 1:
        power = np.abs(complex_channel_data) ** 2
        features.append(np.sum(power, axis=0))  # Total power
        features.append(np.argmax(power, axis=0))  # Dominant path delay

    return np.concatenate([f.flatten() for f in features])


# Transform your data
X_geometric = np.array([extract_geometric_features(channel) for channel in complex_channels])


data = np.load('sionna_sample.npy')
plt.imshow(data[:, 0, :, 0, 0])
plt.colorbar()
plt.show()