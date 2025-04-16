for x_batch, y_batch in train_gen.take(1):
    print("X shape:", x_batch.shape)  # (64, 32, 32, 3)
    print("Y shape:", y_batch.shape)  # (64,)
    print("Labels:", y_batch[:10].numpy())
