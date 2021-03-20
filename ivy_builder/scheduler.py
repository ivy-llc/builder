# global
import json
import traceback
import importlib


def _get_attr(path):
    module, attr = path.rsplit('.', 1)
    module = importlib.import_module(module)
    importlib.reload(module)
    return getattr(module, attr)


class SequentialScheduler:

    def __init__(self, schedule_filepath, num_attempts=1):
        self._num_attempts = num_attempts
        self._schedule_filepath = schedule_filepath
        self._completed_tasks = list()
        self._finished = False

    # Private Methods #
    # ----------------#

    def _load_task(self):
        with open(self._schedule_filepath) as file:
            schedule_dict = json.load(file)
        task_name = None
        for item in schedule_dict.keys():
            if item not in self._completed_tasks:
                task_name = item
                break
        if not task_name:
            return
        task_main_str, cmd_line_args_str = tuple(schedule_dict[task_name])
        main = _get_attr(task_main_str)
        print('\n# ' + '-'*(len(task_name)+14) + '#\n'
              '# Running Task ' + task_name + ' #\n'
              '# ' + '-'*(len(task_name)+14) + '#\n')
        self._completed_tasks.append(task_name)
        if cmd_line_args_str:
            return lambda: main(cmd_line_args_str)
        else:
            return lambda: main()

    # Public Methods #
    # ---------------#

    def run(self):
        while not self._finished:
            task_executable = self._load_task()
            attempt_num = 1
            while True:
                if task_executable:
                    # noinspection PyBroadException
                    try:
                        task_executable()
                        break
                    except Exception:
                        print('\nattempt {} of {}'.format(attempt_num, self._num_attempts) + '\n')
                        print('caught exception: \n {}'.format((traceback.format_exc())))
                        if attempt_num == self._num_attempts:
                            print('Skipping {}'.format(task_executable))
                            break
                        else:
                            print('Re-trying {}'.format(task_executable))
                            attempt_num += 1
                            continue
                else:
                    self._finished = True
                    break
        return self._completed_tasks
