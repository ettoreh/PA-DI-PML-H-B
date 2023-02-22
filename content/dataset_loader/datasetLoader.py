from .loaders import cifar10 


class DatasetLoader():
    """
    class that allows to load dataset just using the name of one
    """
    def __init__(
        self, dataset_name, val_ratio=0.05, batch_size=16, num_workers=4,
        pin_memory=False 
        ) -> None:
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_list = {
            'cifar10': cifar10,
        }
        self.dataset = self.dataset_list[self.dataset_name]
        self.classes = self.dataset.classes
        pass
    
    def get_train_valid_loader(self):
        return self.dataset.trainset_loader(
            self.batch_size, self.val_ratio, num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
    
    def get_test_loader(self):
        return self.dataset.testset_loader(
            self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
    


if __name__ == "__main__":
    import sys 
    
    dataset_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    
    Dataset = DatasetLoader(dataset_name=dataset_name, batch_size=batch_size)
    print(Dataset.trainset_loader())
    