# Load Model Script
fashion_model = CustomFashionNet()
fashion_model.load_state_dict(torch.load('fashion_model_weights.pth'))
fashion_model.eval()

# Evaluation loop
correct_preds = 0
total_imgs = 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        outputs = fashion_model(imgs)
        _, preds = torch.max(outputs.data, 1)
        total_imgs += lbls.size(0)
        correct_preds += (preds == lbls).sum().item()

print(f"Accuracy of the model on the test images: {100 * correct_preds / total_imgs:.2f}%")
