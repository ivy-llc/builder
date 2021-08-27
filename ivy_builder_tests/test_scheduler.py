# global
import os
import io
import time
import json
import threading
from contextlib import redirect_stdout

# local
from ivy_builder.scheduler import SequentialScheduler
from ivy_builder_tests import helpers

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def test_sequential_scheduler(dev_str, call):
    schedule_filepath = os.path.join(THIS_DIR, 'schedule.json')
    scheduler = SequentialScheduler(schedule_filepath)
    helpers.remove_dirs()
    scheduler_len = len(scheduler.run())
    helpers.remove_dirs()
    assert scheduler_len == 2


def test_sequential_scheduler_with_exception(dev_str, call):
    schedule_filepath = os.path.join(THIS_DIR, 'schedule_with_exception.json')
    scheduler = SequentialScheduler(schedule_filepath, num_attempts=2)
    helpers.remove_dirs()
    scheduler_len = len(scheduler.run())
    helpers.remove_dirs()
    assert scheduler_len == 1


def test_sequential_scheduler_with_dynamic_schedule_file_edit(dev_str, call):
    schedule_filepath = os.path.join(THIS_DIR, 'schedule.json')
    scheduler = SequentialScheduler(schedule_filepath)
    helpers.remove_dirs()

    with open(schedule_filepath) as file:
        original_schedule_dict = json.load(file)
    new_schedule_dict = dict(**{'minimal_again': ['demos.simple_example.main', '']},
                             **original_schedule_dict)

    success = [False]

    def write_to_file():
        with open(schedule_filepath, 'w') as schedule_file:
            json.dump(new_schedule_dict, schedule_file, indent=2, separators=(',', ': '))

    def run_scheduler(success_list):
        time.sleep(0.1)
        num_completed_tasks = len(scheduler.run())
        print('found length: {}'.format(num_completed_tasks))
        if num_completed_tasks == 3:
            print('returning true!')
            success_list[0] = True
        else:
            raise Exception('Expected 3 tasks to run, but actually ' + str(num_completed_tasks) + ' ran.')

    file_write_thread = threading.Thread(target=write_to_file)
    schedule_thread = threading.Thread(target=run_scheduler, args=(success,))
    schedule_thread.start()
    file_write_thread.start()
    schedule_thread.join()
    file_write_thread.join()

    with open(schedule_filepath, 'w') as file:
        json.dump(original_schedule_dict, file, indent=2, separators=(',', ': '))

    helpers.remove_dirs()
    assert success[0]


def test_sequential_scheduler_with_dynamic_source_code_edit(dev_str, call):
    schedule_filepath = os.path.join(THIS_DIR, 'schedule.json')
    scheduler = SequentialScheduler(schedule_filepath)
    helpers.remove_dirs()

    src_filepath = os.path.relpath(os.path.join(THIS_DIR, '../demos/full_example.py'))
    with open(src_filepath, 'r') as file:
        original_src_lines = file.readlines()
    msg_to_add = '"Finished complete example!"'
    new_src_lines = original_src_lines[0:287] + ['    print(' + msg_to_add + ')\n'] + original_src_lines[287:]

    success = [False]

    def modify_src_code_file():
        time.sleep(0.1)
        with open(src_filepath, 'w') as src_file:
            src_file.writelines(new_src_lines)

    def run_scheduler(success_list):
        f = io.StringIO()
        with redirect_stdout(f):
            scheduler.run()
        if msg_to_add[1:-1] in f.getvalue():
            success_list[0] = True
        else:
            raise Exception('Expected ' + msg_to_add + ' printed in stdout, but it did not print.')

    schedule_thread = threading.Thread(target=run_scheduler, args=(success,))
    file_write_thread = threading.Thread(target=modify_src_code_file)
    schedule_thread.start()
    file_write_thread.start()
    schedule_thread.join()
    file_write_thread.join()

    with open(src_filepath, 'w') as file:
        file.writelines(original_src_lines)

    helpers.remove_dirs()
    assert success[0]
