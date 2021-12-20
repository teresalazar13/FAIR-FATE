import tensorflow as tf
import tensorflow_federated as tff

from code.tensorflow.models import model_fn


class FederatedLearningClientSide:
    def __init__(self, type_fairness, federated_train_data, input_layer_shape):
        type_fairness = type_fairness
        federated_train_data = federated_train_data
        input_layer_shape = input_layer_shape

        tf_dataset_type = tff.SequenceType(federated_train_data[0].element_spec)

        @tff.tf_computation
        def server_init():
            model = model_fn(federated_train_data, input_layer_shape)
            return model.trainable_variables

        model_weights_type = server_init.type_signature.result

        @tf.function
        def binary_cross_entropy(y_true, y_pred):
            loss = tf.keras.losses.BinaryCrossentropy()
            return loss(y_true, y_pred)

        @tf.function
        def binary_cross_entropy_reweighting(y_true, y_pred, sample_weight):
            loss = tf.keras.losses.BinaryCrossentropy()
            return loss(y_true, y_pred, sample_weight)

        @tf.function
        def client_update(model, dataset, server_weights, client_optimizer):
            client_weights = model.trainable_variables
            tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)

            for batch in dataset:
                with tf.GradientTape() as tape:
                    outputs = model._model._keras_model(batch['x'], training=True)

                    loss = 0
                    if type_fairness == False:
                        loss = binary_cross_entropy(batch['y'], outputs)
                    elif type_fairness == "GR":
                        loss = binary_cross_entropy_reweighting(
                            batch['y'], outputs, batch['reweighting_weights_global']
                        )
                    elif type_fairness == "LR":
                        loss = binary_cross_entropy_reweighting(
                            batch['y'], outputs, batch['reweighting_weights_local']
                        )
                    else:
                        exit(1)

                grads = tape.gradient(loss, client_weights)
                grads_and_vars = zip(grads, client_weights)

                client_optimizer.apply_gradients(grads_and_vars)

            return client_weights

        @tff.federated_computation
        def initialize_fn():
            return tff.federated_value(server_init(), tff.SERVER)

        @tff.tf_computation(tf_dataset_type, model_weights_type)
        def client_update_fn(tf_dataset, server_weights):
            model = model_fn(federated_train_data, input_layer_shape)
            client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

            return client_update(model, tf_dataset, server_weights, client_optimizer)

        federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
        federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

        @tff.federated_computation(federated_server_type, federated_dataset_type)
        def next_fn(server_weights, federated_dataset):
            server_weights_at_client = tff.federated_broadcast(server_weights)
            client_weights = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))

            return server_weights, client_weights

        self.initialize = initialize_fn
        self.next = next_fn
