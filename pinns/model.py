import tensorflow as tf
import time


# Method for creating the model
def net(inputs, layers, activation, outputs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((inputs,)))
    if layers:
        if type(activation) is not list:
            for i in layers:
                model.add(tf.keras.layers.Dense(units=i, activation=activation))
        else:
            for i in activation:
                for j in layers:
                    model.add(tf.keras.layers.Dense(units=j, activation=i))
    model.add(tf.keras.layers.Dense(units=outputs))
    return model


# Method for training
def train(model, geometry, pde, ic_list, epochs, learning_rate):
    print("\nTraining starts...\n")
    train_start = time.process_time()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for i in range(epochs + 1):
        with tf.GradientTape() as tape_model:

            ic_total_error = 0
            for ic in ic_list:
                x_ic = ic[0]
                y_ic_true = ic[1]
                der_index = ic[2]
                component = ic[3]
                # Calculate total ic error
                with tf.GradientTape(persistent=True) as tape_ic:
                    tape_ic.watch(x_ic)
                    y_ic_original = model(x_ic, training=True)
                    y_ic_derivative = y_ic_original[:, component:component + 1]
                    if len(der_index) > 1:
                        # If we have multiple derivatives, they need to be calculated inside tape_ic,
                        # so it can watch them.
                        for j in der_index:
                            y_ic_derivative = tape_ic.gradient(y_ic_derivative, x_ic)[:, j:j + 1]
                if len(der_index) == 1:
                    # If we have only one derivative, we calculate it outside of tape
                    y_ic_derivative = tape_ic.gradient(y_ic_derivative, x_ic)[:, der_index[0]:der_index[0] + 1]
                ic_error = tf.math.square(y_ic_derivative - y_ic_true(tape_ic, x_ic, y_ic_original))

                if len(ic_error.shape) == 2:
                    # In case of multiple input variables reduce their errors to one number
                    ic_error = tf.math.reduce_mean(ic_error)
                ic_total_error = ic_total_error + ic_error
                del tape_ic

            # Fix the form of total ic error
            if ic_total_error.shape == ():
                ic_total_error = tf.reshape(ic_total_error, (1,))

            # Calculate pde error
            with tf.GradientTape(persistent=True) as tape_pde:
                y = model(geometry, training=True)
                domain_error = pde(tape_pde, geometry, y)
            del tape_pde

            # Fix the form of pde error
            domain_total_error = 0
            if type(domain_error) is list:
                for losses in domain_error:
                    domain_total_error = domain_total_error + tf.math.reduce_mean(tf.math.square(losses), axis=0)
            else:
                domain_total_error = tf.math.reduce_mean(tf.math.square(domain_error), axis=0)

            total_error = domain_total_error + ic_total_error

            if i % 1000 == 0:
                print('Epoch: {}\tTotal Loss = {:.2e}\tPDE Loss = {:.2e}\tBC Loss = {:.2e}'.format(
                    i, total_error.numpy()[0], domain_total_error.numpy()[0], ic_total_error.numpy()[0]))

        model_update_gradients = tape_model.gradient(total_error, model.trainable_variables)
        optimizer.apply_gradients(zip(model_update_gradients, model.trainable_variables))
        if i == epochs and i % 1000 != 0:
            print(
                'Epoch: {}\tTotal Loss = {:.2e}\tPDE Loss = {:.2e}\tBC Loss = {:.2e}'.format(
                    i, total_error.numpy()[0], domain_total_error.numpy()[0], ic_total_error.numpy()[0]))
        del tape_model
    train_end = time.process_time() - train_start
    print("\nTraining took", train_end, "s")
