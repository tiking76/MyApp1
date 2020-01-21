"""this is a test code """
import coremltools

path = 'MNIST.h5'

coreml_model = coremltools.converters.keras.convert(path,
                                                    input_names='image',
                                                    output_names='image',
                                                    class_labels='label.txt')
coreml_model.save('number.mlmodel')
