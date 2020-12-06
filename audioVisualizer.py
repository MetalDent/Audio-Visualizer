import librosa              # for audio data
import numpy as np
import pygame               # for pygame animations
from pygame.locals import *
from OpenGL.GL import *     # OpenGL
from OpenGL.GLU import *

def clamp(min_value, max_value, value):
    if value < min_value:
        return min_value

    if value > max_value:
        return max_value
    
    return value

def get_decibel(spectrogram, target_time, time_index_ratio, freq, frequencies_index_ratio):
    return spectrogram[int(freq * frequencies_index_ratio)][int(target_time * time_index_ratio)]

class AudioBar:

    def __init__(self, x, y, freq, color, width=50, min_height=10, max_height=100, min_decibel=-80, max_decibel=0):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height)/(self.max_decibel - self.min_decibel)

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height)/0.1
        self.height += speed * dt
        self.height = clamp(self.min_height, self.max_height, self.height)

    def up_render(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, 400 - self.max_height, self.width, self.height))
        
    def down_render(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y + self.max_height - self.height, self.width, self.height))

def main():
    filename = "sample/sample2.wav"

    time_series, sample_rate = librosa.load(filename)  # getting audio info
    duration = librosa.core.get_duration(y=time_series, sr=sample_rate)

    stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048*4))  # getting a matrix which contains amplitudes acc to freq and time
    
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)  # converting the matrix to decibel matrix

    freqs = librosa.core.fft_frequencies(n_fft=2048*4)  # getting an array of freq

    # getting an array of time periodic
    times = librosa.core.frames_to_time(np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048*4)

    time_index_ratio = len(times)/times[len(times) - 1]

    frequencies_index_ratio = len(freqs)/freqs[len(freqs)-1]

    screen_w = 800      # screen width
    screen_h = 600      # screen height
    pygame.init()       
    display = (screen_h, screen_w)
    screen = pygame.display.set_mode([screen_w, screen_h])      # setup the drawing window
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0, 0, -5)

    bars = []
    frequencies = np.arange(100, 8000, 100)
    r = len(frequencies)
    width = screen_w/r
    x = (screen_w - width*r)/2

    for c in frequencies:
        bars.append(AudioBar(x, 200, c, (2*c%255, c%55, c%200), max_height=400, width=width))
        x += width

    t = pygame.time.get_ticks()
    getTicksLastFrame = t

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play(0)

    # run until quit
    running = True
    while running:
        t = pygame.time.get_ticks()
        deltaTime = (t - getTicksLastFrame) / 1000
        getTicksLastFrame = t

        # quit the window?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((t%2, t%55, t%55))    # background

        for b in bars:
            b.update(deltaTime, get_decibel(spectrogram, pygame.mixer.music.get_pos()/1000.0, time_index_ratio, b.freq, frequencies_index_ratio))
            b.up_render(screen)
            b.down_render(screen)

        pygame.display.flip()   # flip the display

    pygame.quit()   
    
if __name__ == '__main__':
    main()
