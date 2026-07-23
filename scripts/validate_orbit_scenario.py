"""Generic entry point for validating an ephemeris cache and task catalog.

The older ``validate_real_scenario.py`` name remains as a compatibility alias
for deployments that already call it.
"""

from validate_real_scenario import main


if __name__ == "__main__":
    main()
