from distutils.core import setup

setup(name='BPtools',
      version='0.324',
      author='Oliver Rakos',
      author_email='rakos.oliver@mail.bme.hu',
      packages=['BPtools', 'BPtools.core', 'BPtools.metrics', 'BPtools.trainer', 'BPtools.trainer.connectors',
                'BPtools.utils']
      # scripts=['scripts/trainer_vae.py']
      )
