from brec.core.env import KAGGLE, configure_xla_paths

if not KAGGLE:
    configure_xla_paths()


if __name__ == '__main__':
    import argparse

    # main()
