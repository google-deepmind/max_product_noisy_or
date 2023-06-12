# Copyright 2023 DeepMind Technologies Limited.
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
# limitations under the License.

"""Setup for mp_noisy_or package."""

import setuptools


if __name__ == '__main__':
  setuptools.setup(
      name='mp_noisy_or',
      version='0.0.1',
      packages=setuptools.find_packages(),
      license='Apache 2.0',
      author='DeepMind',
      description=(
          'Code for the ICML 2023 paper "Learning noisy-OR Bayesian Networks'
          ' with Max-Product Belief Propagation"'
      ),
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author_email='adedieu@google.com',
      requires_python='>=3.10',
  )
