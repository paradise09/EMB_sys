#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/gpio.h>

/* Meta Information */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Emb_Sys_6 GNU/Linux");
MODULE_DESCRIPTION("Button driver");

/* Variables for device and device class */
static dev_t my_device_nr;
static struct class *my_class;
static struct cdev my_device;

#define DRIVER_NAME "my_button"
#define DRIVER_CLASS "ButtonModuleClass"

/* Read data out of the buffer */
static ssize_t driver_read(struct file *File, char *user_buffer, size_t count, loff_t *offs) {
	int to_copy, not_copied, delta;
	char tmp[2];


	/* Get amount of data to copy */
	to_copy = min(count, sizeof(tmp));

	/* Read value of button */
	tmp[0] = gpio_get_value(22) + '0';
    tmp[1] = gpio_get_value(27) + '0';

	/* Copy data to user */
	not_copied = copy_to_user(user_buffer, &tmp, to_copy);

	/* Calculate data */
	delta = to_copy - not_copied;

	return delta;
}

/* open the driver */
static int driver_open(struct inode *device_file, struct file *instance) {
	printk("segment_button - open was called!\n");
	return 0;
}

/* close the driver */
static int driver_close(struct inode *device_file, struct file *instance) {
	printk("segment_button - close was called!\n");
	return 0;
}

static struct file_operations fops = {
	.owner = THIS_MODULE,
	.open = driver_open,
	.release = driver_close,
	.read = driver_read
};

/* module is loaded into kernel */
static int __init ModuleInit(void) {
    printk("Hello, Kernel!\n");

    /* Allocate a device nr */
	if( alloc_chrdev_region(&my_device_nr, 0, 1, DRIVER_NAME) < 0) {
		printk("Device Nr. could not be allocated!\n");
		return -1;
	}
	printk("read_write - Device Nr. Major: %d, Minor: %d was registered!\n", my_device_nr >> 20, my_device_nr && 0xfffff);

	/* Create device class */
	if((my_class = class_create(THIS_MODULE, DRIVER_CLASS)) == NULL) {
		printk("Device class can not e created!\n");
		goto ClassError;
	}

	/* create device file */
	if(device_create(my_class, NULL, my_device_nr, NULL, DRIVER_NAME) == NULL) {
		printk("Can not create device file!\n");
		goto FileError;
	}

    /* Initialize device file */
	cdev_init(&my_device, &fops);

	/* Regisering device to kernel */
	if(cdev_add(&my_device, my_device_nr, 1) == -1) {
		printk("Registering of device to kernel failed!\n");
		goto AddError;
	}

    /* GPIO 22 init */
	if(gpio_request(22, "rpi-gpio-22")) {
		printk("Can not allocate GPIO 22\n");
		goto AddError;
	}

	/* Set GPIO 22 direction */
	if(gpio_direction_input(22)) {
		printk("Can not set GPIO 22 to input!\n");
		goto Gpio17Error;
	}

    /* GPIO 27 init */
	if(gpio_request(27, "rpi-gpio-27")) {
		printk("Can not allocate GPIO 27\n");
		goto AddError;
	}

	/* Set GPIO 27 direction */
	if(gpio_direction_input(27)) {
		printk("Can not set GPIO 27 to input!\n");
		goto Gpio27Error;
	}

    return 0;

Gpio17Error:
    gpio_free(22);
Gpio27Error:
    gpio_free(27);
AddError:
	device_destroy(my_class, my_device_nr);
FileError:
	class_destroy(my_class);
ClassError:
	unregister_chrdev_region(my_device_nr, 1);
	return -1;
}

static void __exit ModuleExit(void) {
    gpio_free(22);
    gpio_free(27);
    cdev_del(&my_device);
	device_destroy(my_class, my_device_nr);
	class_destroy(my_class);
	unregister_chrdev_region(my_device_nr, 1);
	printk("Goodbye, Kernel\n");
}

module_init(ModuleInit);
module_exit(ModuleExit);