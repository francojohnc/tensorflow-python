from matplotlib import pyplot as plt


# expect numpy data
def show_image(image, label):
    plt.subplots_adjust(0, 0, 1, 1)
    plt.ylabel(label)
    plt.imshow(image)
    plt.show()
