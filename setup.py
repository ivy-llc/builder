# lint as: python3
# Copyright 2021 The Ivy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License..
# ==============================================================================
from distutils.core import setup
import setuptools

setup(name='ivy-builder',
      version='1.1.6',
      author='Ivy Team',
      author_email='ivydl.team@gmail.com',
      description='Build custom Ivy training tasks with clear, hierarchical and robust user specifications.\n',
      long_description="""Build custom Ivy training tasks with clear, hierarchical and robust user specifications.\n""",
      long_description_content_type='text/markdown',
      url='https://ivy-dl.org/ivy',
      project_urls={
            'Docs': 'https://ivy-dl.org/builder/',
            'Source': 'https://github.com/ivy-dl/builder',
      },
      packages=setuptools.find_packages(),
      install_requires=['ivy-core'],
      scripts=['scripts/print_json_args.py', 'scripts/remove_checkpoints.py', 'scripts/reset_to_defaults.sh',
               'scripts/format_dataset_containers.py'],
      classifiers=['License :: OSI Approved :: Apache Software License'],
      license='Apache 2.0'
      )
