# Isolated Azure test VM

`test-vm.json` creates a dedicated Linux test VM and its own network, public IP,
disk, and daily deallocation schedule. Deploy it only into a new resource group
for this project. It does not refer to other projects' resources or credentials.

The initial configuration is East US 2, `Standard_D4as_v7` (4 vCPUs, 16 GiB), a
128-GiB Premium LRS OS disk, and the pinned Canonical Ubuntu 26.04 server image.
The image has no separately billed software plan. This is a serial integration
test host; it does not provide the resources for the full benchmark release.

## Funding and shutdown

Before deploying or restarting, verify that the selected subscription belongs
to the intended billing profile, its credit grant is active and covers ordinary
Azure compute, and its estimated remaining balance includes pending eligible
charges. A subscription name, resource tag, Foundry deployment, budget alert, or
shutdown schedule does not establish or enforce credit-only billing.

The September 28, 2026 public rates checked for this configuration were $0.182
per VM hour and $17.92/month for the P10 LRS disk. Public IPv4 and any metered
network traffic are additional. Disk and IP charges continue after deallocation.
Keep the initial shutdown within six hours of deployment, and deallocate the VM
explicitly when work finishes. Verify the schedule exists after deployment:
ARM deployments can fail after creating some billable resources.

Azure startup sponsorship applies credits to eligible consumption, but an
account with its spending limit off can charge a payment method after credit
exhaustion or expiry. Other projects may consume the same credit pool. Do not
describe this template as a hard spending cap, and do not deploy when grant
coverage or sufficient remaining credit is unverified.

## Deployment inputs

Prepare an ignored, private ARM parameters file with:

- `sshPublicKey`: a dedicated SSH public key for this host.
- `sshSourceCidr`: the operator's validated public IPv4 address followed by `/32`.
- `cloudInit`: the base64-encoded bytes of `cloud-init.yaml`.
- `shutdownUtc`: the daily UTC deallocation time in `HHmm` format.

The Azure CLI administrator must have permission to register
`Microsoft.DevTestLab` for scheduled deallocation. Validate the template and
inspect `az deployment group what-if` in the new resource group before creating
the resources. Keep the parameter file, private key, billing responses and
deployment output outside Git.

## Runtime handoff

Cloud-init creates `bugbash` with UID/GID 1000 and no supplementary groups, plus
the separate SSH/sudo administrator `tasteadmin` with UID 1001. Verify those
identities and successful `cloud-init status --wait` before executing tests.
Only the administrator runs bounded Docker/systemd fixtures; unit tests run as
`bugbash` without access to the Docker socket. No model keys are installed.

Install the Docker Buildx plugin and nftables explicitly for benchmark image
builds and Harbor's network restrictions. On the current Ubuntu host,
`docker-buildx` 0.30.1-0ubuntu1 passed a network-disabled scratch-image build,
load and artifact-copy check; its temporary image/container were removed.
The presence of nftables does not substitute for Harbor's live egress check.

Rebuild the worker and Harbor virtual environments separately from recorded
package versions. Transfer source and audit evidence with hash verification.
Recreate containers and use new owner tokens: old process IDs, cgroups, broker
ledgers, and launch credentials are evidence, not resumable runtime state.

After handoff, verify host versions, dependencies, source hashes, serial unit
checks, and a real Harbor fixture with independent cleanup. Preserve the old
host until the handoff is verified; deleting this project's processes does not
cancel the old provider's VM rental or authorize deleting other projects.
