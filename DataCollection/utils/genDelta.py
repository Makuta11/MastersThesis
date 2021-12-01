import wave

data = [128 for i in range(8200)] # zeroes
for i in range(80):
    data[i] = 255

data = bytes(data) # convert to bytes

with open(r'Sounds/delta.wav', 'wb') as file:
    f = wave.open(file)
    f.setnchannels(1) # mono
    f.setsampwidth(1) 
    f.setframerate(8200) # standard sample rate
    f.writeframes(data)
