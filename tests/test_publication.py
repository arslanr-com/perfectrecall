"""Release contracts: fresh installation, legacy settings, and one provider."""
import json
import os
from pathlib import Path

import pytest
import yaml

from perfectrecall.install import configure_hermes


def test_installer_creates_first_provider_without_old_package(tmp_path):
    home=tmp_path/'empty-profile'
    result=configure_hermes(home)
    assert yaml.safe_load((home/'config.yaml').read_text())=={'memory':{'provider':'perfectrecall'}}
    assert result['backup'] is None
    assert not (home/'mnemosyne').exists()  # Installation never rewrites memory.


def test_installer_preserves_settings_and_disables_duplicate_hooks(tmp_path):
    home=tmp_path/'profile';home.mkdir()
    original='model: example\nmemory:\n  provider: mnemosyne\n  mnemosyne:\n    profile_isolation: true\n    sync_roles: [user]\n'
    (home/'config.yaml').write_text(original)
    plugin=home/'plugins/mnemosyne';plugin.mkdir(parents=True)
    (plugin/'marker.txt').write_text('existing plugin')
    result=configure_hermes(home)
    config=yaml.safe_load((home/'config.yaml').read_text())
    assert config['model']=='example'
    assert config['memory']['mnemosyne']['profile_isolation'] is True
    assert Path(result['backup']).read_text()==original
    assert not plugin.exists()
    assert (Path(result['disabled_legacy_plugins'][0])/'marker.txt').read_text()=='existing plugin'
    assert configure_hermes(home)['disabled_legacy_plugins']==[]


def test_bad_config_is_never_overwritten(tmp_path):
    (tmp_path/'config.yaml').write_text('memory: []\n')
    with pytest.raises(ValueError):configure_hermes(tmp_path)
    assert (tmp_path/'config.yaml').read_text()=='memory: []\n'


def test_new_settings_override_legacy_keys_without_losing_others(tmp_path):
    from mnemosyne.hermes_config import read_hermes_config_key
    (tmp_path/'config.yaml').write_text('memory:\n  mnemosyne:\n    auto_sleep: false\n    sync_roles: [user]\n  perfectrecall:\n    sync_roles: []\n')
    assert read_hermes_config_key(str(tmp_path),'auto_sleep') is False
    assert read_hermes_config_key(str(tmp_path),'sync_roles')==[]


def test_provider_registers_itself_before_class_discovery():
    from perfectrecall.hermes import register,PerfectRecallMemoryProvider
    class Context:
        def register_memory_provider(self,provider):self.provider=provider
        def register_cli_command(self,**kwargs):self.command=kwargs['name']
    context=Context();register(context)
    assert isinstance(context.provider,PerfectRecallMemoryProvider)
    assert context.provider.name==context.command=='perfectrecall'


def test_hermes_and_mcp_use_the_same_caller_guidance():
    from mnemosyne.tool_schemas import RECALL_SCHEMA as mcp
    from hermes_memory_provider import RECALL_SCHEMA as hermes
    assert mcp['description']==hermes['description']
    assert 'SHORT' in hermes['description']
    assert 'baseline mode' not in hermes['description']


def test_no_production_vector_modules_or_dependencies():
    import importlib.util
    for module in ('embeddings','binary_vectors','polyphonic_recall','query_cache'):
        assert importlib.util.find_spec('mnemosyne.core.'+module) is None
    root=Path(__file__).resolve().parents[1]
    pyproject=(root/'pyproject.toml').read_text()
    for dependency in ('fastembed','sqlite-vec','onnxruntime','sentence-transformers'):
        assert dependency not in pyproject
