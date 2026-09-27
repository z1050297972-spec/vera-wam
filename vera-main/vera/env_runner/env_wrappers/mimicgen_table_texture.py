"""Dark-wood table texture for robosuite's TableArena, without forking robosuite.

The released mimicgen WAN planner was trained on dark-wood renders; stock robosuite's
white ceramic table is out-of-distribution for it. Rather than shipping a patched
robosuite, we swap the table texture in memory: TableArena parses its MJCF and
resolves every asset ``file`` attribute to an absolute path inside its ``__init__``
(MujocoXML.resolve_asset_dependency), so a post-``__init__`` rewrite of the
``tex-ceramic`` texture node is enough. ``dark-wood.png`` already ships in stock
robosuite's ``models/assets/textures`` dir, so no asset file needs to be added.

Call :func:`apply_dark_wood_table` once before constructing any robosuite/mimicgen
env (e.g. at the top of MimicgenRunner.setup_env).
"""

import logging
import os

logger = logging.getLogger(__name__)

# stock table_arena.xml names the tabletop texture "tex-ceramic"; we keep the name
# and only repoint the file so the material binding is untouched.
_TABLE_TEXTURE_NAME = "tex-ceramic"
_DARK_WOOD_FILENAME = "dark-wood.png"


def apply_dark_wood_table() -> None:
    """Patch robosuite's TableArena so the tabletop renders with dark-wood.

    Idempotent; safe to call once per process before env construction. Covers
    TableArena subclasses too (they run TableArena.__init__ via super())."""
    import robosuite
    from robosuite.models.arenas import table_arena as _table_arena_mod

    cls = _table_arena_mod.TableArena
    if getattr(cls, "_vera_dark_wood_patched", False):
        return  # already applied

    tex_path = os.path.join(robosuite.models.assets_root, "textures", _DARK_WOOD_FILENAME)
    if not os.path.isfile(tex_path):
        # don't hard-fail env setup over a cosmetic patch; the planner will just see OOD renders
        logger.warning("dark-wood texture not found at %s; keeping stock table texture", tex_path)
        return

    orig_init = cls.__init__

    def _dark_wood_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # asset file attrs are absolute by now; repoint the tabletop cube texture
        node = self.asset.find(f"./texture[@name='{_TABLE_TEXTURE_NAME}']")
        if node is not None:
            node.set("file", tex_path)
        else:
            logger.warning("texture '%s' not found in %s; table texture left unchanged",
                           _TABLE_TEXTURE_NAME, type(self).__name__)

    cls.__init__ = _dark_wood_init
    cls._vera_dark_wood_patched = True
    logger.info("TableArena patched: tabletop texture -> %s", tex_path)
