

class DefaultLogger:
    def __init__(*args, **kwargs):
        pass

    def log_metric(self, a, b):
        print(f'({a}: {b})')

    def log_parameters(self, a):
        for k, v, in a.items():
            print(f'{k:>30}: {v}')

    def log_message(self, msg):
        print(msg)

