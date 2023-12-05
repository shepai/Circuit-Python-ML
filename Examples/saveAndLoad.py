import board
import busio
import digitalio
import adafruit_sdcard
import storage
import ulab.numpy as np


# Set up SPI bus
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)

# Configure CS pin
cs = digitalio.DigitalInOut(board.FLASH_CS)  # Replace D10 with the actual CS pin used
sdcard = adafruit_sdcard.SDCard(spi, cs)

# Mount the SD card
vfs = storage.VfsFat(sdcard)
storage.mount(vfs, "/sd")

data=normal(0,0.5,(100,20))

ae=AutoEncoder(20,10)
ae.train(data,10,0.001,n_prints=1)
ae.save("/sd/testModel.json")

