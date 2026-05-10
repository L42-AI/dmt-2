from dotenv import load_dotenv
load_dotenv()

from pipeline import Pipeline

train_predictions, val_predictions, test_predictions, train_loss, val_loss, test_loss = Pipeline().run('baseline')

print(f"Train Loss: {train_loss}")
print(f"Validation Loss: {val_loss}")
print(f"Test Loss: {test_loss}")