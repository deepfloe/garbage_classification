def file_path(grayscale_conversion, rescale_factor, folder_name):
    folder = 'npy_files/'
    if grayscale_conversion:
        file_path = folder + 'grayscale_' + str(rescale_factor) + '_' + folder_name
    else:
        file_path = folder + 'rgb_' + str(rescale_factor) + '_' + folder_name

    return file_path