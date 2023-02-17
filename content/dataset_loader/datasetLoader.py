from loaders import cifar10 


class DatasetLoader():
    """
    class that allows to load dataset just using the name of one
    """
    def __init__(self, dataset_name, batch_size=16) -> None:
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_list = {
            'cifar10': cifar10,
        }
        self.dataset = self.dataset_list[self.dataset_name]
        pass
    
    def trainset_loader(self):
        return self.dataset.trainset_loader(self.batch_size)
    
    def testset_loader(self):
        return self.dataset.testset_loader(self.batch_size)
    


if __name__ == "__main__":
    import sys 
    
    dataset_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    
    Dataset = DatasetLoader(dataset_name=dataset_name, batch_size=batch_size)
    print(Dataset.trainset_loader())
    