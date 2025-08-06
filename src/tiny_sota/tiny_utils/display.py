class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ORANGE = '\x1B[38;5;216;4m'
    WARNING = '\033[93m'
    NICE = '\x1B[38;5;216;1m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

printGreen = lambda term: print(f"{bcolors.GREEN}{term}{bcolors.ENDC}")
printCyan = lambda term: print(f"{bcolors.CYAN}{term}{bcolors.ENDC}")
printWarn = lambda term: print(f"{bcolors.WARNING}{term}{bcolors.ENDC}")
printNice = lambda term: print(f"{bcolors.NICE}{term}{bcolors.ENDC}")
printBlue = lambda term: print(f"{bcolors.BLUE}{term}{bcolors.ENDC}")
printOrange = lambda term: print(f"{bcolors.ORANGE}{term}{bcolors.ENDC}")

def count_parameters(model, only_trainable=True):
    total_params = None
    if only_trainable:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params

def table_parameters(model, only_trainable=True):
    def get_max_len_name(model):
        names = [n for n, _ in model.named_parameters()]
        return len(max(names, key=len))
    s_l = get_max_len_name(model)
    sum_fn = lambda l: sum(i.numel() for i in l.parameters())
    printNice(f"{model.__class__.__name__:_^{s_l}}|{'Parameters':_^25}|{'Set To Train':_^25}".upper())
    for n, l in model.named_children():
        if len(list(l.parameters())) and next(l.parameters()).requires_grad is not None:
            printOrange(f"{n:.^{s_l}}|{sum_fn(l):.^25,}|{'layer':.^25}".upper())
            for n_i, p in l.named_parameters():
                n_it = f"{n}.{n_i}"
                if p.requires_grad == True:
                    printGreen(f"{n_it:.^{s_l}}|{p.numel():.^25,}|{'True':.^25}")
                else:
                    printBlue(f"{n_it:.^{s_l}}|{p.numel():.^25,}|{'False':.^25}")
        else:
            printWarn(f"{n:.^{s_l}}|{'NA':.^25}|{'NA':.^25}".upper())

def name_all_layers(model, only_trainable=True):
    if only_trainable:
        layers = [n for n, c in model.named_children() if len(list(c.parameters()))]
    else:
        layers = [n for n, _ in model.named_children()]
    return layers

def name_subset_layers(model, trainable=True):
    layers = [(n, c) for n, c in model.named_children() if len(list(c.parameters()))]
    layers = [n for n, c in layers if next(c.parameters()).requires_grad == trainable]
    return layers
     