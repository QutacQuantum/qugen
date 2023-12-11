import sys
import os
import subprocess
import platform
import json


# test that the correct environment is activated
print("Hello!")

print("Trying to import the libraries needed to run QUGEN...")
try:
    import pennylane, jax, glob, re, abc, pathlib, typing, numpy, pandas, tqdm, time, hashlib, cma, pickle
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    print("Successfull!")
    print("\n")
except ModuleNotFoundError:
    print("The required libraries could not be imported. ")
    print("Make sure you have the correct virtual environment activated and set it up accourding to README.md. ")
    print("Then restart this script. ")
    print("Stopping ...")
    print("\n")
    sys.exit()

print("Trying to import the QUGEN specific modules...")
try:
    from qugen.main.generator.continuous_qcbm_model_handler import ContinuousQCBMModelHandler
    from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler
    from qugen.main.generator.discrete_qgan_model_handler import DiscreteQGANModelHandler
    from qugen.main.generator.discrete_qcbm_model_handler import DiscreteQCBMModelHandler
    from qugen.main.data.data_handler import load_data
    
    print("Successfull!")
    print("\n")
except ModuleNotFoundError: 
    print("Unable to load code for models. Please ensure that the repository has been set up correctly and try again.")
    print("Stopping... ")
    print("\n")
    sys.exit()


try:
    import questionary
except ModuleNotFoundError:
    if input("Press enter to 'pip install questionary' (type no to cancel) ") == "":
        subprocess.run(["pip", "install", "questionary"])
        import questionary
    else:
        sys.exit()


# creating autocomplete #
# Dictionary to store completion options for each question
completion_options = {
    "main": ["load model", "create model", "train model", "show model info", "test experiment", "add dataset", "stop"],
    "data": ["Continuous", "Discrete"],
    "model": ["QGAN", "QCBM"],
    "circuit": ["copula", "standard"],
    "transformation": ["pit", "minmax"],
    "tests": ["KL Loss", "Sample", "other", "save (select to have generated visualisations saved to the disk)", "stop"],
    "tests_save": ["KL Loss", "Sample", "other", "don't save (select to have generated visualisations only shown, not saved)", "stop"],
    # Add more questions and options as needed
}

# Function to set the completer based on the current question
def get_choices(question_key):
    return completion_options.get(question_key, [])

# setting the working directory so that this script independent of where it is called from #
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Set the current working directory to the script's directory
os.chdir(script_dir)

def isfloat(var):
    if var.replace(".", "").isnumeric() and var.count(".") in [0,1]:
        return True
    else:
        return False

def integer_input(var_name, help_text="no help text implemented yet"):
    var = input(f"Which {var_name} should be used (Enter 'help' for help)? ")
    if var == "help":
        print(help_text)
    while not var.isnumeric():
        if var != "help":
            print("Please enter an integer.")
        var = input(f"Which {var_name} should be used (Enter 'help' for help)? ")
        if var == "help":
            print(help_text)
    return int(var)

def float_input(var_name, help_text="no help text implemented yet"):
    var = input(f"Which {var_name} should be used (Enter 'help' for help)? ")
    if var == "help":
        print(help_text)
    while not isfloat(var):
        if var != "help":
            print("Please enter a float. Seperate by '.' not by ','")
        var = input(f"Which {var_name} should be used (Enter 'help' for help)? ")
        if var == "help":
            print(help_text)
    return float(var)

def get_epoch(model_name):
    path = os.path.join(experiment_path,model_name)
    
    if len([f for f in os.listdir(os.path.join(experiment_path, model_name)) if any([f.endswith(x) for x in [".npy",".pickle"]])]) == 0:
        return 0
    
    def split_datafile_names(filename):
        filename = filename.split(".")[0]
        filename = filename.split("=")[1]
        filename = int(filename)
        return filename
    return max(list(map(split_datafile_names,[f for f in os.listdir(path) if any([f.endswith(x) for x in [".npy", ".pickle"]]) and not f.startswith("reverse_lookup")])))

def get_model(chosen_model_info):
    datatype = chosen_model_info["circuit_type"]
    modeltype = chosen_model_info["model_type"]
    match [datatype, modeltype]:
        case ["continuous", "QCBM"]:
            model = ContinuousQCBMModelHandler()
        case ["continuous", "QGAN"]:
            model = ContinuousQGANModelHandler()
        case ["discrete", "QCBM"]:
            model = DiscreteQCBMModelHandler()
        case ["discrete", "QGAN"]:
            model = DiscreteQGANModelHandler()
        case _:
            print("Model Type could not be established. Stopping without loading model.")
            return None
    return model

def get_data_sets():
   return list(map(lambda x: x.split(".npy")[0],[f for f in os.listdir("apps/logistics/training_data") if f.endswith(".npy")]))

############## code specifying the user interactions #########

## functions for the differen choices in the main menu ##

def load_model(model_info):
    chosen_model, model = None, None
    if len(model_info.keys()) == 0:
        print("No models found. Please Create a model first.")
    else:
        # create model
        chosen_model = questionary.select("Which model do you want to choose?", choices=model_info.keys()).ask()
        model = get_model(model_info[chosen_model])
        if model is None: # pass error through if it arises.
            return None, None
        # get num of epochs
        last_epoch = get_epoch(chosen_model)
        # change model directory so that reload functions correctly as it is hard coded to work if the file is in the apps/logistics directory
        os.chdir(os.path.join(script_dir,"apps/logistics"))
        # reload model params
        model = model.reload(chosen_model, epoch=last_epoch)
        os.chdir(script_dir) # change back to the default for this script
    return chosen_model, model 



def create_model(model_info):
    # parameter choice
    datatype = questionary.select("Which type of data should the model use?", choices=get_choices("data")).ask()
    modeltype = questionary.select("Which model type should be used?", choices=get_choices("model")).ask()
    dataset = questionary.select("Which dataset should be used?", choices=get_data_sets()).ask()
    if datatype != "Continuous" or modeltype != "QCBM":
        transformation = questionary.select("Which transformation should the model use?", choices=get_choices("transformation")).ask()
    if datatype == "Discrete":
        circuit_type = questionary.select("Which circuit type should the model use?", choices=get_choices("circuit")).ask()
    
    circuit_depth = integer_input("circut depth")
    if modeltype == "QCBM":
        initial_sigma = float_input("initial sigma")
    
    if datatype == "Discrete":
        n_qubits = integer_input("number of qubits", "The number of QBits available to the model.")
        n_registers = integer_input("number of registers")
    
    # name choice
    chosen_model_name = input("What should the model be named (Cancel the Creation with 'Cancel')? ")
    if chosen_model_name == "Cancel":
        return None, None
    while chosen_model_name in os.listdir(experiment_path):
        if input("Name already taken. Override? [y/n] ") in ["y","Y"]:
            break
        else:
            chosen_model_name = input("What should the model be named (Cancel the Creation with 'Cancel')? ")
    while input(f"Confirm model name {chosen_model_name} ") != "":
        chosen_model_name = input("What should the model be named (Cancel the Creation with 'Cancel')? ")
        if chosen_model_name == "Cancel":
            return None, None
    
    print("Creating model...")
    # change model directory so that reload functions correctly as it is hard coded to work if the file is in the apps/logistics directory
    os.chdir(os.path.join(script_dir,"apps/logistics"))
    data_set_path =  f"training_data/{dataset}"
    data, _ = load_data(data_set_path)
    if datatype == "Continuous":
        n_qubits = data.shape[1]
        
    # choose and build model according to the specifications
    match [datatype, modeltype]:
        case ["Continuous", "QCBM"]:
            model = ContinuousQCBMModelHandler()
            
            model.build(
                model_name= chosen_model_name,
                data_set=dataset, 
                n_qubits=n_qubits, 
                circuit_depth=circuit_depth, 
                initial_sigma=initial_sigma,
                true_name=True
            )
        case ["Continuous", "QGAN"]:
            model = ContinuousQGANModelHandler()
            
            model.build(
                model_name= chosen_model_name,
                data_set=dataset, 
                n_qubits=n_qubits, 
                circuit_depth=circuit_depth, 
                transformation=transformation,
                true_name=True
            )
        case ["Discrete", "QCBM"]:
            model = DiscreteQCBMModelHandler()
            
            model.build(
                model_name= chosen_model_name,
                data_set=dataset, 
                n_qubits=n_qubits, 
                n_registers=n_registers,
                circuit_depth=circuit_depth,
                initial_sigma=0.01,
                circuit_type=circuit_type,
                transformation=transformation,
                hot_start_path="",
                true_name=True
            )
        case ["Discrete", "QGAN"]:
            model = DiscreteQGANModelHandler()
            
            model.build(
                model_name= chosen_model_name,
                data_set=dataset, 
                n_qubits=n_qubits, 
                n_registers=n_registers, 
                circuit_depth=circuit_depth,
                transformation=transformation,
                circuit_type=circuit_type,
                true_name=True
            )
    os.chdir(script_dir) # change back to the default for this script
           
    print("Done. \n")
    
    model_info[chosen_model_name] = model.metadata
    
    return chosen_model_name, model

def train_model(model, model_info):
    # get necessary specifications of the model
    datatype, modeltype = model_info["circuit_type"], model_info["model_type"]
    # get training parameters
    n_epochs = integer_input("number of epochs")
    batch_size = integer_input("batch size")
    if modeltype == "QGAN":
        initial_learning_rate_generator = float_input("initial generator learning rate") 
        initial_learning_rate_discriminator = float_input("initial discriminator learning rate")
    elif modeltype == "QCBM":
        hist_samples = integer_input("number of histogram samples")
    # change model directory so that reload functions correctly as it is hard coded to work if the file is in the apps/logistics directory
    os.chdir(os.path.join(script_dir,"apps/logistics")) 
    data_set_path = os.path.join("training_data",model_info["data_set"])
    data, _ = load_data(data_set_path)
    # train the model according to modeltype
    match [datatype, modeltype]:
        case ["continuous", "QCBM"]:
            print("Continuous", "QCBM")
            model.train(data,
                n_epochs=n_epochs,
                batch_size=batch_size, 
                hist_samples=hist_samples
            )
        case ["continuous", "QGAN"]:
            print("Continuous", "QGAN")
            model.train(
                data, 
                n_epochs=n_epochs, 
                initial_learning_rate_generator=initial_learning_rate_generator, 
                initial_learning_rate_discriminator=initial_learning_rate_discriminator, 
                batch_size=batch_size
            )
        case ["discrete", "QCBM"]:
            print("Discrete", "QCBM")
            model.train(
                data,
                n_epochs=n_epochs,
                batch_size=batch_size,
                hist_samples=hist_samples,
            )
        case ["discrete", "QGAN"]:
            print("Discrete", "QGAN")
            model.train(
                data, 
                n_epochs=n_epochs, 
                initial_learning_rate_generator=initial_learning_rate_generator, 
                initial_learning_rate_discriminator=initial_learning_rate_discriminator,
                batch_size=batch_size,
            )
    os.chdir(script_dir) # change back to the default for this script
    print("Training successfull!")

def add_dataset(): 
    print("Your dataset should be a single npy file.")
    print("It should contain a NxM matrix where N is the number of samples you have and M the dimensionality of your data.")
    input("Please place your file before you press enter. ")
    dataset_choices = [f for f in list(os.listdir()) if f.endswith(".npy")]
    while len(dataset_choices) == 0:
        input("No npy files found. Please place your file before you press enter. ")
        dataset_choices = [f for f in list(os.listdir()) if f.endswith(".npy")]
    dataset_name = questionary.select("Which is your dataset?", choices=dataset_choices).ask()
    if platform.system() == "Windows":
        result = subprocess.run(["copy", dataset_name, "apps\\logistics\\training_data"])
    else:
        result = subprocess.run(["cp", dataset_name, "apps/logistics/training_data"])
    if result.returncode == 0:
        print(f"Successfully copied {dataset_name}!")
    else:
        print(f"Error incured. \nError code: {result.returncode}. \nError message: {result}")
    
def test_experiment(model_info, model):
    # TODO: Ideen: Show as animation how sampling changes with training; Loss landscape projected in 2D
    # load dataset
    dataset = model_info["data_set"] # TODO gives Error
    # change model directory so that reload functions correctly as it is hard coded to work if the file is in the apps/logistics directory
    os.chdir(os.path.join(script_dir,"apps/logistics"))
    data_set_path =  f"training_data/{dataset}"
    data, _ = load_data(data_set_path)
    os.chdir(script_dir) # change back to the default for this script
    savemode = False
    while True:
        if savemode:
            chosen_option = questionary.select("What kind of test or visualisation do you want to do?", choices=get_choices("tests")).ask()
        else:
            chosen_option = questionary.select("What kind of test or visualisation do you want to do?", choices=get_choices("tests_save")).ask()
        match chosen_option:
            case "KL Loss":
                print("Evaluating model...")
                evaluation_df = model.evaluate(data)

                # find the model with the minimum Kullbach-Liebler divergence:

                minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
                minimum_kl_calculated = minimum_kl_data["kl_original_space"]
                print(f"{minimum_kl_calculated=}")
                print(evaluation_df)
                # plt.evaluation_df
            case "Sample":
                print("Not yet fully implemented")
                
                # get data dimension
                dimension = data.shape[1]
            
                # generate samples from a trained model:
                number_samples = integer_input("number of samples shown")
                compare = ""
                while compare not in ["y", "n"]:
                    compare = input("Should the original dataset be displayed to compare [y/n]? ")
                compare = dict({"y": True, "n": False})[compare]
                samples = model.predict(number_samples)
                
                # plot 2D samples:
                title = input("Plot Title: ")
                match dimension:
                    case 1:
                        print("not yet implemented")
                    case 2:
                        if compare:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                            fig.title(title)
                            # Scatter plot for subplot 1
                            ax1.scatter(samples[:, 0], samples[:, 1], color="blue")
                            ax1.set_title("Generated data")
                            
                            # Scatter plot for subplot 2
                            ax2.scatter(data[:, 0], data[:, 1], color="red")
                            ax2.set_title("Real Data")
                            if savemode:
                                plt.savefig(title.replace(" ","_")+".png")
                            plt.show()
                        else:
                            plt.scatter(samples[:, 0], samples[:, 1])
                            plt.title(title)
                            if savemode:
                                plt.savefig(title.replace(" ","_")+".png")
                            plt.show()
                    case 3:
                        if compare:
                            fig = plt.figure(figsize=(12, 6))
                            fig.title(title)
                            # Scatter plot for subplot 1
                            ax1.scatter(samples[:, 0], samples[:, 1], color="blue")
                            ax1.set_title("Generated data")
                            
                            # Scatter plot for subplot 2
                            ax2.scatter(data[:, 0], data[:, 1], color="red")
                            ax2.set_title("Real Data")
                            if savemode:
                                plt.savefig(title.replace(" ","_")+".png")
                            plt.show()
                        else:
                            plt.scatter(samples[:, 0], samples[:, 1])
                            plt.title(title)
                            if savemode:
                                plt.savefig(title.replace(" ","_")+".png")
                            plt.show()
                        pass # , projection='3d'
                    case _:
                        if dimension<1:
                            print("please enter a number that is greater or equal to 1.")
                        else:
                            print("Projections into lower dimensions from higher dimensions are not yet implemented.")
            case "other":
                print("Implement further visualisation stuff!")
            case "save (select to have generated visualisations saved to the disk)":
                savemode = True
                print("Savemode on.")
            case "don't save (select to have generated visualisations only shown, not saved)":
                savemode = False
                print("Savemode off.")
            case "stop":
                break
            case _:
                print("Input not recognized. Exiting...")
                break



# short explainer

with open("description") as file:
    for i in file.readlines():
        print(i)

# main menu with choices

# static variables
experiment_path = "apps/logistics/experiments"

# saved info
chosen_model_name = None
chosen_model = None
loaded_models = [f for f in os.listdir("apps/logistics/experiments") if not any(f.endswith(ext) for ext in [".json", ".py"])]
model_info = {}
for model in loaded_models:
    path = os.path.join(experiment_path,model,"meta.json")
    with open(path) as file:
        model_info[model] = json.load(file)


while True:
    print("\n")
        
    if chosen_model_name is None:
        print("You have no model selected. Select or create a model before proceeding. ")
    else:
        print(f"You currently have {chosen_model_name} selected.")
        if len([f for f in os.listdir(os.path.join(experiment_path, chosen_model_name)) if f.endswith(".npy")]) > 0:
            print("The model is trained.") 
        else:
            print("The model is not trained.")
            
    chosen_option = questionary.select("What do you want to do?", choices=get_choices("main")).ask()
    match chosen_option:
        case "load model":
            chosen_model_name, chosen_model = load_model(model_info)
        case "create model":
            chosen_model_name, chosen_model = create_model(model_info)
        case "train model":
            if chosen_model_name is not None:
                train_model(chosen_model, model_info[chosen_model_name])
            else:
                print("Choose a model first")
        case "show model info":
            if chosen_model_name is None:
                print("You need to load a model first.")
            else:
                #TODO more info pretty
                for key in model_info[chosen_model_name]:
                    print(f"{key}:   {model_info[chosen_model_name][key]}")
        case "add dataset":
            add_dataset()
        case "test experiment":
            test_experiment(model_info[chosen_model_name], chosen_model)
        case "stop":
            if True:#input("You have no unsaved data. Press Enter to continue... ") == "": # if there is the possibility of unsaved data in the future, warn here.
                print("Stopping ...")
                # here any neccesities for graceful stopping can be added in the future:
                sys.exit()
            else:
                print("Canceled. ")
        case _:
            print("Your input was not recognized. Please make sure your input matches one of the options exactly including capitalization. ")

