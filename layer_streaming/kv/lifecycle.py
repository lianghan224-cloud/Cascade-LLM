"""Frozen Page Descriptor lifecycle transition contract."""

from .errors import KVLifecycleError
from .types import PageState


PAGE_STATE_TRANSITIONS = {
    PageState.FREE: {PageState.ALLOCATED},
    PageState.ALLOCATED: {
        PageState.ACTIVE,
        PageState.COPYING,
        PageState.RELEASING,
    },
    PageState.ACTIVE: {
        PageState.ACTIVE,
        PageState.SEALED,
        PageState.RELEASING,
    },
    PageState.SEALED: {
        PageState.ACTIVE,
        PageState.SEALED,
        PageState.SHARED,
        PageState.MIGRATING,
        PageState.EVICT_PENDING,
        PageState.RELEASING,
    },
    PageState.SHARED: {
        PageState.SHARED,
        PageState.SEALED,
        PageState.MIGRATING,
        PageState.EVICT_PENDING,
        PageState.RELEASING,
    },
    PageState.COPYING: {PageState.ACTIVE, PageState.RELEASING},
    PageState.MIGRATING: {PageState.SEALED, PageState.SHARED, PageState.EVICT_PENDING},
    PageState.EVICT_PENDING: {PageState.RELEASING, PageState.SEALED, PageState.SHARED},
    PageState.RELEASING: {PageState.FREE},
}


def transition_page(descriptor, target):
    target = PageState(target)
    allowed = PAGE_STATE_TRANSITIONS.get(descriptor.state, set())
    if target not in allowed:
        raise KVLifecycleError(
            "invalid page transition {} -> {}".format(
                descriptor.state.value,
                target.value,
            )
        )
    descriptor.state = target
    return descriptor
