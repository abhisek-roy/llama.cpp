# About These Notes

Reviewed against upstream `192067b72` and the private-fork merge resolution of 2026-08-27.

This folder is a personal architecture notebook for understanding llama.cpp. It is not official upstream documentation.

The repo README remains the high-level project entry point:

- [llama.cpp README](../README.md)

These notes are meant to explain how parts of the system work from an architecture point of view, with diagrams, code maps, and practical experiments. Each major topic should live as a chapter under `notes/`.

Pages distinguish the current implementation from historical measurements. A review marker identifies the upstream revision used for the latest architecture audit; benchmark sections keep their original context when results have not yet been rerun.

Current chapters:

- [RPC architecture](rpc/index.md)
