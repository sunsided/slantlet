# Releasing

Releases are cut locally with [cargo-release](https://github.com/crate-ci/cargo-release)
and published by CI. Pushing a `vX.Y.Z` tag runs `.github/workflows/release.yml`,
which checks that the tag matches `Cargo.toml`, runs the tests, publishes to
crates.io via Trusted Publishing, and creates a GitHub Release with the
packaged `.crate` attached.

## Cutting a release

From an up-to-date `main` with a clean tree:

```sh
cargo release patch            # dry run: shows what would happen
cargo release patch --execute  # bump, commit "Release X.Y.Z", tag vX.Y.Z, push
```

Use `minor` or `major` instead of `patch` as needed. `release.toml` allows
releases only from `main` and leaves publishing to CI (`publish = false`).
The `release` environment requires approval before the publish job runs.

## One-time setup

1. **First publish is manual.** crates.io Trusted Publishing can only be
   configured for a crate that already exists, so publish the first version
   with an API token:

   ```sh
   cargo login            # token from https://crates.io/settings/tokens
   cargo publish
   git tag v0.1.0 && git push origin v0.1.0
   ```

   The tag push starts the release workflow; reject its pending `release`
   deployment (the version is already on crates.io) and create the first
   GitHub Release by hand:

   ```sh
   cargo package
   gh release create v0.1.0 --verify-tag --generate-notes target/package/slantlet-0.1.0.crate
   ```

   Revoke the API token afterwards.

2. **Configure the trusted publisher** on crates.io: crate settings →
   *Trusted Publishing* → *Add* → GitHub, with
   - owner: `sunsided`
   - repository: `slantlet`
   - workflow filename: `release.yml`
   - environment: `release`

3. **GitHub environment** `release` (already created): requires approval
   by a reviewer and only accepts deployments from `v*` tags. Adjust under
   *Settings → Environments → release*.
