# Hyperparameters of the baseline models
batch_size = 16
alpha = 1e-3
epochs = 20

# Define a function to execute the model
def modelEXE(model_dir,history_dir, model, X_train, y_train, X_val,y_val,epochs=epochs,batch_size=batch_size):
  """
  This function reads the model and history from the directory and returns the model and history if it exists, 
  otherwise it compiles the model and returns the model and history.
  """
  if not os.path.exists(model_dir):

    # Compile the model with cross validation
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ## Set the checkpoint
    checkpoint = ModelCheckpoint(f"{model_dir}".split('.')[0]+".keras",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)

    ##Set the early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='auto', restore_best_weights=True)

    ## Instantiate the TensorBoard callback
    # tensorboard = TensorBoard(log_dir = 'logs')

    ## Set the learning rate scheduler  
    reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,mode='auto',verbose=1)
                              
    # Data Augmentation
    img_datagen.fit(X_train)

    # Training of the baseline model
    history = model.fit(img_datagen.flow(tf.cast(X_train, tf.float32), np.array(pd.get_dummies(y_train)),shuffle=True), validation_data=(tf.cast(X_val, tf.float32), np.array(pd.get_dummies(y_val))), epochs =epochs, verbose=1, batch_size=batch_size,  callbacks=[early_stopping, checkpoint,reduce_lr])

    # Show the model summary
    # model.summary()

    # Store the history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_dir, index=False)
    return model, history_df

  else:
    # load the history
    history_df = pd.read_csv(history_dir)
    model_loaded= tf.keras.models.load_model(model_dir)
    # evaluate the model
    model_loaded.evaluate(tf.cast(X_val, tf.float32), np.array(pd.get_dummies(y_val)))
    return model_loaded, history_df

# Define a function to plot the history
def plot_history(history_df):
    plt.figure(figsize=(15, 5))
    plt.plot(history_df['accuracy'], label='Train Accuracy',color='red')
    plt.plot(history_df['val_accuracy'], '-.', label='Validation Accuracy',color='red')
    plt.plot(history_df['loss'], label='Train Loss',color='blue')
    plt.plot(history_df['val_loss'],'-.', label='Validation Loss',color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')
    plt.legend()
    plt.grid()
    plt.show()