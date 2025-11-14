import json
import os
import re
from typing import Tuple, Any, Dict, Optional


class ConfigManager:
    """
    Load base and application-specific JSON configurations with environment variable interpolation.

    Parameters
    ----------
    config_path : str, optional
        Filesystem path to the general configuration JSON. Defaults to the value of the ``CONFIG_PATH`` environment variable.
    app_config_path : str, optional
        Filesystem path to the application configuration JSON. Defaults to the value of the ``APP_CONFIG_PATH`` environment variable.
    app_id : str, optional
        Logical application identifier used to select the configuration subset. Default is ``"cardiology_protocols"``.
    """
    _config_path: str #: str : Path to the general configuration file.
    _app_config_path: str #: str : Path to the application configuration file.
    _app_id: str #: str : Selected application identifier.
    _config: Dict[str, Any] #: Dict[str, Any] : Parsed application configurations indexed by application id.
    _general_config: Dict[str, Any] #: Dict[str, Any] : Parsed general configuration.
    _app_config: Dict[str, Any] #: Dict[str, Any] : Merged configuration for the selected ``_app_id``.
    def __init__(self,
                 config_path: str = os.getenv("CONFIG_PATH"),
                 app_config_path: Optional[str] = None,
                 app_id: str = "cardiology_protocols"):
        self._config_path = config_path
        self._app_config_path = app_config_path
        self._app_id = app_id
        self._config, self._general_config = self._load_config()
        self._app_config = self._get_app_config()

    def _load_config(self) -> Tuple[Dict[str, Any], Dict[Any, Any] | None]:
        """
        Read and parse the general and application configuration files, interpolating
        environment variables written as ``${VAR}``.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            A tuple ``(config_json, app_config_json)`` with the parsed dictionaries.

        Raises
        ------
        FileNotFoundError
            If either configuration file cannot be found.
        ValueError
            If either file contains invalid JSON.
        """
        try:
            with open(self._config_path, "r") as config_file:
                raw_config_json = config_file.read()
            raw_app_config_json = None
            if self._app_config_path is not None:
                with open(self._app_config_path, "r") as app_config_file:
                    raw_app_config_json = app_config_file.read()

            def replace_env_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, f"<MISSING:{var_name}>")
            interpolated_json = re.sub(r"\$\{(\w+)\}", replace_env_var, raw_config_json)
            config_json = json.loads(interpolated_json)
            app_config_json = None
            if raw_app_config_json is not None:
                interpolated_app_json = re.sub(r"\$\{(\w+)\}", replace_env_var, raw_app_config_json)
                app_config_json = json.loads(interpolated_app_json)
            return config_json, app_config_json
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self._config_path} or {self._app_config_path}")
        except json.JSONDecodeError:
            raise ValueError(
                f"Invalid JSON format in configuration file at {self._config_path} or {self._app_config_path}")

    def _get_app_config(self) -> Dict[str, Any]:
        """
        Build the effective configuration for ``self._app_id`` by merging the application-specific configuration with the general one.

        .. rubric:: Notes

        Merging uses the dict union operator ``config | general_config``;
        on key conflicts, values from ``general_config`` override those from ``config``.

        Returns
        -------
        Dict[str, Any]
            The merged application configuration.

        Raises
        ------
        ValueError
            If no configuration exists for the selected ``_app_id``.
        """
        general_config = self._general_config.get(self._app_id) if self._general_config is not None else dict()
        config = self._config.get(self._app_id)
        if not config:
            raise ValueError(f"No configuration found for application: {self._app_id}")
        app_config = general_config | config
        return app_config