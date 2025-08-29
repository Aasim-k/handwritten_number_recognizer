import joblib
from sklearn import svm
import mnist_loader  # assuming your custom loader

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    
    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    
    # save model
    joblib.dump(clf, "svm_mnist_model.pkl")
    print("Model saved to svm_mnist_model.pkl")
    
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("Baseline classifier using an SVM.")
    print("%s of %s values correct." % (num_correct, len(test_data[1])))


if __name__ == "__main__":
    svm_baseline()
