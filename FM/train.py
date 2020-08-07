import tensorflow as tf
import numpy as np

def l2_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def train_step(mod, x, y_true, lr,
               loss_f = l2_loss):
    # Calculate current loss and record gradients
    with tf.GradientTape() as t:
        cur_loss = loss_f(y_true=y_true,
                          y_pred=mod(x))

    # Get the gradients and assign to model
    db, dw, dv = t.gradient(cur_loss, [mod.b, mod.w, mod.v])
    mod.b.assign_sub(lr * db)
    mod.w.assign_sub(lr * dw)
    mod.v.assign_sub(lr * dv)

    return cur_loss

def run_model(model, x, y, epochs=200) :
    bs, ws, vs, losses = [], [], [], []

    for e in range(epochs):

        cur_loss = train_step(mod=model, x=x,
                              y_true=y,
                              lr=0.0025,
                              loss_f=l2_loss)

        # Logging
        losses.append(cur_loss)
        bs.append(model.b.numpy()[0])
        # Just logging changes in first terms here
        ws.append(model.w.numpy()[0])
        vs.append(model.v.numpy()[0])

        # Sometimes plot loss
        if (e > 1) & (e % 10 == 0):
            print(f"Epoch {e + 1} / {epochs}: Loss: {cur_loss.numpy()}, last change: {np.abs(losses[-1] - losses[-2])}")

        # Early stop?
        if e > 10:
            if np.mean(losses[-10:-1]) < 0.1:
                print(f"Early stopping, loss={cur_loss.numpy()}")

                return model
    return model
