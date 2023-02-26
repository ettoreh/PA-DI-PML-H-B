

def get_step_logs(i, N, loss, ber):
    steplog = " - Step [{}/{}] \t Loss: {:.4f} \t BER: {:.4f}".format(
        i+1, N, loss, ber)
    print(steplog)
    return None

def get_epoch_logs(epoch, num_epoch, train_loss, val_loss, val_acc, time, BER):
    trainlog = "Epoch [{}/{}], \t Train loss: {:.4f}, \t Validation".format(
        epoch+1, num_epoch, train_loss)
    trainlog += " loss: {:.4f}, \t Validation acc: {:.4f}, \t Time: {}".format(
        val_loss, val_acc, time)
    trainlog += ", \t BER: {:.3f}".format(BER)
    print(trainlog)
    return None   
    
def get_eval_logs(acc, acc_per_classes):
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    print(f'***')
    for classname, acc in acc_per_classes.items():
        print(f'Accuracy for class: {classname:5s} is {acc:.2f} %')
    return None

def get_prune_logs(layer_sparsity, global_sparsity, layers):
    for i, layer in enumerate(layers):
        print("Sparsity in {}: {:.2f}%".format(layer[1], layer_sparsity[i]))
    print("Global sparsity: {:.2f}%".format(global_sparsity))
    return None