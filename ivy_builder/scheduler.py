# global
import os
import sys
import ivy
import json
import traceback
import importlib
import subprocess

SHARED_JSS = dict()


def _get_attr(path):
    module, attr = path.rsplit('.', 1)
    module = importlib.import_module(module)
    importlib.reload(module)
    return getattr(module, attr)


class SequentialScheduler:

    def __init__(self, schedule_filepath, num_attempts=1, set_experiment_name=False):
        self._num_attempts = num_attempts
        self._set_experiment_name = set_experiment_name
        self._schedule_filepath = schedule_filepath
        self._completed_tasks = list()
        self._finished = False

    # Private Methods #
    # ----------------#

    def _load_task(self):
        global SHARED_JSS
        with open(self._schedule_filepath) as file:
            file_str = file.read()
            file_str_split = file_str.split('spec_dict(')
            spec_dicts = [spec_dict.split(')')[0] for spec_dict in file_str_split[1:]]
            spec_dicts_formatted = [spec_dict.replace('{', '__curley_open__').replace('}', '__curley_close))').replace(
                '[', '__square_open__').replace(']', '__square_close__').replace(':', '__colon__').replace(
                ',', '__comma__').replace('\\"', '\'').replace(' ', '') for spec_dict in spec_dicts]
            file_str_formatted = file_str
            for spec_dict, spec_dict_formatted in zip(spec_dicts, spec_dicts_formatted):
                file_str_formatted = file_str_formatted.replace(spec_dict, spec_dict_formatted)
            schedule_dict = json.loads(file_str_formatted)
        task_name = None
        if 'jss' in schedule_dict.keys():
            if list(schedule_dict.keys())[0] != 'jss':
                raise Exception('jss must be the first key at top of the schedule.json file')
            del schedule_dict['jss']
            spec_dicts_formatted.pop(0)
            jss_spec_dict_str = spec_dicts.pop(0).replace('\\"', '"').replace(' ', '')
            SHARED_JSS = json.loads(jss_spec_dict_str)

        for item in schedule_dict.keys():
            if item not in self._completed_tasks:
                task_name = item
                break
        if not task_name:
            return None, None
        task_main_str, cmd_line_args_str = tuple(schedule_dict[task_name])
        if self._set_experiment_name:
            cmd_line_args_str += ' -en ' + task_name
        print('\n# ' + '-'*(len(task_name)+14) + '#\n'
              '# Running Task ' + task_name + ' #\n'
              '# ' + '-'*(len(task_name)+14) + '#\n')
        self._completed_tasks.append(task_name)
        if cmd_line_args_str:
            for spec_dict, spec_dict_formatted in zip(spec_dicts, spec_dicts_formatted):
                cmd_line_args_str =\
                    cmd_line_args_str.replace(spec_dict_formatted, spec_dict.replace('\\"', '"').replace(' ', ''))
            if SHARED_JSS:
                if 'spec_dict' in cmd_line_args_str:
                    existing_spec_dict_str = cmd_line_args_str.split('spec_dict(')[-1].split(')')[0]
                    existing_spec_dict = json.loads(existing_spec_dict_str)
                    combined_spec_dict = ivy.Container.combine(
                        ivy.Container(SHARED_JSS), ivy.Container(existing_spec_dict)).to_dict()
                    combined_spec_dict_str = json.dumps(combined_spec_dict)
                    cmd_line_args_str.replace(existing_spec_dict_str, combined_spec_dict_str)
                else:
                    cmd_line_args_str += ' -jss spec_dict(' + json.dumps(SHARED_JSS).replace(' ', '') + ')'
        cmd_line_args_str = cmd_line_args_str.replace('spec_dict(', "\'").replace(')', "\'")
        return task_main_str, cmd_line_args_str

    # Public Methods #
    # ---------------#

    def run(self):
        while not self._finished:
            task_main_str, cmd_line_args_str = self._load_task()
            attempt_num = 1
            while True:
                if task_main_str:
                    # noinspection PyBroadException
                    try:
                        process = subprocess.Popen(
                            'python3 -m ' + '.'.join(task_main_str.split('.')[:-1]) + ' ' + cmd_line_args_str,
                            stdout=subprocess.PIPE, shell=True, cwd=os.getcwd())
                        for c in iter(lambda: process.stdout.read(1), b''):
                            try:
                                sys.stdout.write(c)
                            except TypeError:
                                break
                        process.wait()
                        break
                    except Exception:
                        print('\nattempt {} of {}'.format(attempt_num, self._num_attempts) + '\n')
                        print('caught exception: \n {}'.format((traceback.format_exc())))
                        if attempt_num == self._num_attempts:
                            print('Skipping {}'.format(task_main_str))
                            break
                        else:
                            print('Re-trying {}'.format(task_main_str))
                            attempt_num += 1
                            continue
                else:
                    self._finished = True
                    break
        return self._completed_tasks
