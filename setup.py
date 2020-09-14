from distutils.core import setup

setup(name='BPtools',
      version='0.1',
      author='Oliver Rakos',
      author_email='rakos.oliver@mail.bme.hu',
      packages=['BPtools','BPtools.core', 'BPtools.trainer', 'BPtools.utils']
      # scripts=['scripts/trainer_vae.py']
      )
