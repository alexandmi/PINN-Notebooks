import tensorflow as tf

J = {}
H = {}


# Method for jacobian derivatives
def jacobian(tape, y, x, i=0, j=0):

    key = (y.ref(), x.ref())

    if y.shape[1] > 1:
        y = y[:, i: i + 1]
    if key not in J:
        y_index = {i: tape.gradient(y, x)}
        J[key] = y_index
    if i not in J[key]:
        J[key][i] = tape.gradient(y, x)

    return J[key][i][:, j: j + 1]


# Method for hessian derivatives
def hessian(tape, y, x, i=0, j=0, component=0):

    key = (y.ref(), x.ref())
    if (key not in J) or (component not in J[key]):
        jacobian(tape, y, x, component, i)

    grad = J[key][component][:, i: i + 1]

    if key not in H:
        x_index = {i: tape.gradient(grad, x)}
        y_index = {component: x_index}
        H[key] = y_index

    if component not in H[key]:
        x_index = {i: tape.gradient(grad, x)}
        H[key][component] = x_index

    if i not in H[key][component]:
        H[key][component][i] = tape.gradient(grad, x)

    return H[key][component][i][:, j: j + 1]


# Method for fixing an input
def reshape_input(ic_input, tensor_size):
    if type(ic_input[0]) is not float and type(ic_input[0]) is not int:
        full_tensor = ic_input[0]
    else:
        temp_vector = tf.fill([tensor_size, 1], tf.cast(ic_input[0], tf.float32))
        temp_variable = tf.Variable(temp_vector, trainable=True, dtype=tf.float32)
        full_tensor = temp_variable

    for i in ic_input[1:]:
        if type(i) is not float and type(i) is not int:
            full_tensor = tf.concat([full_tensor, i], 1)
        else:
            temp_vector = tf.fill([tensor_size, 1], tf.cast(i, tf.float32))
            temp_variable = tf.Variable(temp_vector, trainable=True, dtype=tf.float32)
            full_tensor = tf.concat([full_tensor, temp_variable], 1)

    return full_tensor


# Method for initial conditions
def initial_condition(input, output, derivative=0, der_index=None, component=0):

    if der_index is None:
        der_index = []
    # Input is a list.
    if type(input) is list:
        tensor_found = False
        tensor_size = 0
        # Check the values of the list in case there is a tensor.
        for i in input:
            if type(i) is not float and type(i) is not int:
                tensor_found = True
                tensor_size = i.shape[0]
                break
        if tensor_found:
            # If there is at least one tensor in the list, use the reshape_input method to fix the input.
            x_ic = reshape_input(input, tensor_size)
        else:
            # If there are only numbers in the list, make it a tf.Variable.
            x_ic = tf.Variable([input], trainable=True, dtype=tf.float32)

    # Input is one item
    else:
        if type(input) is not float and type(input) is not int:
            # If the input is a tensor, take it as it is
            x_ic = input
        else:
            # If the input is a number, make it a tf.Variable
            x_ic = tf.Variable([[input]], trainable=True, dtype=tf.float32)
        # If the input is one item, all derivatives
        der_index = [0] * derivative

    if type(output) is not type(reshape_input):
        # If the output is a number, make a method for it
        def y_ic(_tape, _x, _y):
            return output
    else:
        # If the output is a method, keep it as it is
        y_ic = output

    return [x_ic, y_ic, der_index, component]
